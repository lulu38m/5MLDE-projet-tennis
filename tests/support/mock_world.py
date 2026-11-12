"""Monde simulé partagé par les tests : mini-dataset synthétique dont toutes les dates sont connues à l'avance.

- `2019-0001` Alpha Open (ATP 250, 8 joueurs, début 2019-01-07) : QF le 07/01, SF le 10/01, F le 13/01
- `2019-D001` Coupe Davis (début 2019-03-01) : 4 rencontres de poule sur 3 jours (01, 01, 02, 03/03)
- `2020-0002` Beta Open (ATP 250, 4 joueurs, début 2020-01-06) : SF le 06/01, F le 12/01
- joueur 9 : inscrit mais n'a jamais joué (débutant) ; joueur 10 : inconnu
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from mock_source.chaos import ChaosController
from mock_source.clock import SimulatedClock
from mock_source.repository import ReplayRepository
from tennis_mlops.config import MockSourceConfig, PauseConfig
from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.lockfile import DataLock, FileRecord

CONFIG = MockSourceConfig(
    first_round_notice_days=2,
    tournament_notice_days=14,
    fixture_horizon_days=7,
    pauses=(PauseConfig(start=date(2019, 4, 1), end=date(2019, 12, 31), reason="Pause de test"),),
)
SCORES = ["6-4 6-4", "6-3 2-1 RET", "W/O", "7-6(5) 6-7(3) 6-2", "6-1 6-1", "6-4 3-6 7-5", "6-2 6-2"]


def match(
    tid: str, name: str, level: str, draw: int, start: int, num: int, rnd: str, w: int, lo: int, score: str
) -> dict[str, object]:
    return {
        "tourney_id": tid, "tourney_name": name, "surface": "Hard", "draw_size": draw, "tourney_level": level,
        "tourney_date": start, "match_num": num, "score": score, "best_of": 3, "round": rnd, "minutes": 90,
        "winner_id": w, "winner_seed": None, "winner_entry": None, "winner_name": f"Joueur {w}", "winner_hand": "R",
        "winner_ht": 185, "winner_ioc": "FRA", "winner_age": 25.04, "winner_rank": w, "winner_rank_points": 1000 - w,
        "loser_id": lo, "loser_seed": None, "loser_entry": "Q", "loser_name": f"Joueur {lo}", "loser_hand": "L",
        "loser_ht": None, "loser_ioc": "ESP", "loser_age": 30.0, "loser_rank": lo, "loser_rank_points": 1000 - lo,
    }  # fmt: skip


def write_dataset(raw: Path) -> tuple[DatasetCatalog, DataLock]:
    raw.mkdir(parents=True)
    catalog = DatasetCatalog(2019, 2020)
    alpha = [
        ("QF", 1, 1, 8), ("QF", 2, 2, 7), ("QF", 3, 3, 6), ("QF", 4, 4, 5),
        ("SF", 5, 1, 4), ("SF", 6, 2, 3), ("F", 7, 1, 2),
    ]  # fmt: skip
    rows_2019 = [
        match("2019-0001", "Alpha Open", "A", 8, 20190107, n, r, w, lo, SCORES[n - 1]) for r, n, w, lo in alpha
    ]
    davis = [(1, 1, 5), (2, 2, 6), (4, 1, 6), (5, 2, 5)]
    rows_2019 += [match("2019-D001", "Davis Cup", "D", 4, 20190301, n, "RR", w, lo, "6-3 6-3") for n, w, lo in davis]
    beta = [("SF", 1, 1, 4), ("SF", 2, 2, 3), ("F", 3, 2, 1)]
    rows_2020 = [match("2020-0002", "Beta Open", "A", 4, 20200106, n, r, w, lo, "6-4 6-4") for r, n, w, lo in beta]
    pd.DataFrame(rows_2019).to_csv(raw / "atp_matches_2019.csv", index=False)
    pd.DataFrame(rows_2020).to_csv(raw / "atp_matches_2020.csv", index=False)

    players = [
        {"player_id": i, "name_first": "Joueur", "name_last": str(i), "hand": "R", "dob": "19940101", "ioc": "FRA",
         "height": 185, "wikidata_id": None}
        for i in range(1, 10)
    ]  # fmt: skip
    pd.DataFrame(players).to_csv(raw / "atp_players.csv", index=False)

    def rankings(day: int) -> pd.DataFrame:
        return pd.DataFrame(
            {"ranking_date": day, "rank": range(1, 9), "player": range(1, 9), "points": range(900, 100, -100)}
        )

    pd.concat([rankings(20190107), rankings(20190114)]).to_csv(raw / "atp_rankings_10s.csv", index=False)
    rankings(20200106).to_csv(raw / "atp_rankings_20s.csv", index=False)
    rankings(20200113).to_csv(raw / "atp_rankings_current.csv", index=False)
    (raw / "matches_data_dictionary.txt").write_text("dictionnaire")
    (raw / "UPSTREAM_README.md").write_text("readme")
    lock = DataLock.build({}, {name: FileRecord.of(raw / name) for name in catalog.names()})
    return catalog, lock


class FakeTime:
    """Temps réel contrôlé par le test."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds


@dataclass
class MockEnv:
    repository: ReplayRepository
    clock: SimulatedClock
    chaos: ChaosController
    client: TestClient
    raw: Path
    catalog: DatasetCatalog
    lock: DataLock

    def at(self, day: date) -> TestClient:
        self.clock.set(day)
        return self.client


def write_project_config(root: Path, *, replay_start: str = "2019-02-01", state_dir: Path | None = None) -> Path:
    """Projet de test sur le mini-dataset : années 2019-2020, retries sans attente, état dans `state_dir`."""
    from tennis_mlops.config import DEFAULT_CONFIG_PATH

    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    for old, new in [
        ("first_match_year: 2010", "first_match_year: 2019"),
        ("last_match_year: 2026", "last_match_year: 2020"),
        ("bootstrap_start: 2010-01-01", "bootstrap_start: 2019-01-01"),
        ("replay_start: 2015-01-01", f"replay_start: {replay_start}"),
        ("replay_end: 2026-05-31", "replay_end: 2020-12-31"),
        ("retry_delays_s: [1, 2, 4]", "retry_delays_s: [0, 0, 0]"),
        ("drift_min_history: 20", "drift_min_history: 3"),
        ("state_dir: state", f"state_dir: {state_dir or root / 'state'}"),
        ("training_trigger_matches: 200", "training_trigger_matches: 5"),
    ]:
        assert old in text, old
        text = text.replace(old, new)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    path = root / "configs" / "project.yaml"
    path.write_text(text, encoding="utf-8")
    return path
