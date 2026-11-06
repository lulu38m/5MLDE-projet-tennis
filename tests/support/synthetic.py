"""Historique de matchs synthétique au schéma de la table `matches`, pour tester features et entraînement.

Chaque joueur a un niveau caché ; le vainqueur est tiré selon l'écart de niveau. Le classement publié suit ce
niveau avec du bruit : un modèle doit donc faire mieux que le hasard, sans que rien ne soit écrit d'avance.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import numpy.typing as npt
import pandas as pd

SURFACES = ("Hard", "Clay", "Grass")
LEVELS = ("A", "M", "G")


def synthetic_matches(
    start: date = date(2010, 1, 4), weeks: int = 160, matches_per_week: int = 40, players: int = 80, seed: int = 7
) -> pd.DataFrame:
    """Un tournoi par semaine (`matches_per_week` matchs répartis sur 4 jours), dataset_version `synthetique`."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, players)
    rank_order = np.argsort(-(skill + rng.normal(0, 0.3, players)))
    ranks = np.empty(players, dtype=int)
    ranks[rank_order] = np.arange(1, players + 1)
    rows = []
    for week in range(weeks):
        monday = start + timedelta(weeks=week)
        tourney_id = f"{monday.year}-{week:04d}"
        surface, level = SURFACES[week % 3], LEVELS[week % 3]
        for num in range(1, matches_per_week + 1):
            a, b = rng.choice(players, size=2, replace=False)
            p_a = 1 / (1 + np.exp(-(skill[a] - skill[b]) * 1.5))
            winner, loser = (a, b) if rng.random() < p_a else (b, a)
            status = "walkover" if num == matches_per_week and week % 10 == 0 else "completed"
            rows.append({
                "match_id": f"{tourney_id}-{num}", "tourney_id": tourney_id, "tourney_name": f"Open {week}",
                "tourney_level": level, "surface": surface, "draw_size": 64, "best_of": 5 if level == "G" else 3,
                "round": "R32" if num % 4 else "QF", "match_num": num,
                "match_date": monday + timedelta(days=num % 4), "status": status,
                "score": "6-4 6-4" if status == "completed" else "W/O", "minutes": 90,
                **_player("winner", winner, ranks), **_player("loser", loser, ranks),
                "lot_id": tourney_id, "dataset_version": "synthetique", "ingested_at": pd.Timestamp("2026-01-01"),
            })  # fmt: skip
    return pd.DataFrame(rows)


def _player(side: str, index: int, ranks: npt.NDArray[np.int_]) -> dict[str, object]:
    player_id = 100_000 + int(index)
    return {
        f"{side}_id": player_id,
        f"{side}_name": f"Joueur {index}",
        f"{side}_hand": "L" if index % 7 == 0 else "R",
        f"{side}_ht": 170 + index % 30,
        f"{side}_ioc": "FRA",
        f"{side}_age": 20 + index % 15,
        f"{side}_rank": int(ranks[index]),
        f"{side}_rank_points": int(5000 / ranks[index]),
        f"{side}_seed": None,
        f"{side}_entry": None,
    }


class FrameSource:
    """Source de matchs en mémoire (remplace PostgreSQL dans les tests)."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def matches_frame(self) -> pd.DataFrame:
        return self.frame.copy()

    def lots_summary(self) -> dict[str, object]:
        return {"lots_valid": int(self.frame["tourney_id"].nunique()), "lots_quarantined": 0,
                "last_valid_lot": str(self.frame["tourney_id"].iloc[-1])}  # fmt: skip
