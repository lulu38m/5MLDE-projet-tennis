"""Historique des joueurs servi à l'interface : exact, borné à l'historique validé, cohérent avec le modèle."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa

from tennis_mlops.features.builder import PLAYED_STATUSES
from tennis_mlops.serving.history import PlayerHistory, normalize
from tennis_mlops.serving.persistence import ApiStore
from tennis_mlops.serving.schemas import MatchRequest, PlayerRef, PlayerSummary
from tennis_mlops.serving.service import PredictionBatch, UnknownPlayerError
from tests.support.serving import ServingWorld, match_body, serving_world


@pytest.fixture
def world(tmp_path: Path) -> ServingWorld:
    return serving_world(tmp_path).ready()


def _played(frame: pd.DataFrame, player: int) -> pd.DataFrame:
    played = frame[frame["status"].isin(PLAYED_STATUSES)]
    return played[(played["winner_id"] == player) | (played["loser_id"] == player)]


def test_player_stats_match_the_validated_matches(world: ServingWorld) -> None:
    stats = world.service.player_stats(100_001)
    mine = _played(world.frame, 100_001)
    wins = int((mine["winner_id"] == 100_001).sum())
    assert (stats.overall.wins, stats.overall.losses) == (wins, len(mine) - wins)  # forfaits exclus
    assert sum(r.wins + r.losses for r in stats.by_surface) == len(mine)
    assert sum(r.wins + r.losses for r in stats.by_level) == len(mine)
    assert stats.as_of == stats.history_last_day == world.last_day
    last = mine.sort_values(["match_date", "match_num"]).tail(10)
    assert stats.form.results == (last["winner_id"] == 100_001).tolist() and stats.form.window == 10
    assert len(stats.recent) == 15 and stats.recent[0].match_date >= stats.recent[-1].match_date
    assert stats.ranking_history and stats.ranking_history == sorted(stats.ranking_history, key=lambda p: p.date)
    # L'Elo affiché est exactement celui que le modèle reçoit.
    ratings = world.service.features.current.builder.ratings(100_001)  # type: ignore[union-attr]
    assert stats.ratings is not None and stats.ratings.elo == ratings.elo and stats.ratings.form == ratings.form


def test_as_of_never_reaches_past_the_history(world: ServingWorld) -> None:
    early = date(2010, 3, 1)
    stats = world.service.player_stats(100_001, early)
    mine = _played(world.frame, 100_001)
    before = mine[pd.to_datetime(mine["match_date"]).dt.date <= early]
    assert stats.as_of == early and stats.overall.wins + stats.overall.losses == len(before)
    assert all(m.match_date <= early for m in stats.recent) and stats.player.last_match <= early  # type: ignore[operator]
    assert stats.ratings is None  # l'Elo n'est pas historisé : pas d'Elo « d'aujourd'hui » pour une date passée
    future = world.service.player_stats(100_001, world.last_day + timedelta(days=400))
    assert future.as_of == world.last_day  # une date future est ramenée à la dernière journée connue


def test_unknown_player(world: ServingWorld) -> None:
    with pytest.raises(UnknownPlayerError, match="débutant"):
        world.service.player_stats(999_999)


def test_head_to_head_agrees_with_the_model_feature(world: ServingWorld) -> None:
    builder = world.service.features.current.builder  # type: ignore[union-attr]
    pairs = [(a, b) for a in range(100_000, 100_010) for b in range(100_000, 100_010) if a != b]
    for p1, p2 in pairs:
        h2h = world.service.head_to_head(p1, p2)
        assert (h2h.p1_wins, h2h.p2_wins) == (builder._h2h.get((p1, p2), 0), builder._h2h.get((p2, p1), 0))
        assert h2h.total == len(h2h.meetings) == sum(s.p1_wins + s.p2_wins for s in h2h.by_surface)
    h2h = world.service.head_to_head(100_001, 100_002)
    feature = world.client.post("/predict", json=match_body(100_001, 100_002)).json()["features"]["h2h_diff"]
    assert feature == h2h.p1_wins - h2h.p2_wins


def test_search(world: ServingWorld) -> None:
    found = world.service.search_players("joueur 1", limit=50).players
    assert found and all("1" in (p.name or "") for p in found)
    assert [p.id for p in world.service.search_players("100007").players] == [100_007]
    ranked = world.service.search_players(None, limit=5).players
    assert [p.rank for p in ranked] == sorted(p.rank for p in ranked if p.rank is not None)
    assert world.service.search_players("zzz").count == 0


def test_search_ignores_accents_and_case() -> None:
    day = date(2015, 1, 1)
    profiles = {1: PlayerSummary(id=1, name="Gaël Monfils", rank=20, matches=10, last_match=day),
                2: PlayerSummary(id=2, name="Gael Monfils Jr", rank=None, matches=1, last_match=day)}  # fmt: skip
    history = PlayerHistory(pd.DataFrame(), profiles)
    assert [p.id for p in history.search(profiles, "GAEL monf", 10)] == [1, 2]
    assert normalize("Ćorić") == "coric"
    assert history.h2h(PlayerRef(id=1, name=None), PlayerRef(id=2, name=None)).total == 0


def test_fixture_predictions_follow_their_matches(world: ServingWorld) -> None:
    played = world.frame[world.frame["status"] == "completed"].iloc[-1]
    walkover = world.frame[world.frame["status"] == "walkover"].iloc[-1]
    future = world.last_day + timedelta(days=3)
    # Trois prédictions faites « avant » : un match joué depuis, un forfait, un match encore à venir.
    for row in (played, walkover):
        body = match_body(int(row["winner_id"]), int(row["loser_id"]), match_id=row["match_id"],
                          tourney_name="Open prédit")  # fmt: skip
        world.service.record(*_predicted(world, body))
    world.service.record(*_predicted(world, match_body(match_id="2099-1-1", match_date=future.isoformat())))
    world.service.record(*_predicted(world, match_body(match_id="2099-1-2"), client="ui"))  # pas du scoring
    world.flush()
    response = world.service.fixture_predictions(days=400)
    by_id = {f.match_id: f for f in response.fixtures}
    assert set(by_id) == {played["match_id"], walkover["match_id"], "2099-1-1"}
    done = by_id[played["match_id"]]
    assert done.status == "played" and done.actual_winner_id == int(played["winner_id"])
    assert done.correct == (done.predicted_winner_id == done.actual_winner_id) and done.score == played["score"]
    assert done.winner_probability == done.player1.win_probability  # player1 = vrai vainqueur ici
    assert done.tourney_name == "Open prédit"
    assert by_id[walkover["match_id"]].status == "walkover" and by_id[walkover["match_id"]].correct is None
    assert by_id["2099-1-1"].status == "upcoming" and by_id["2099-1-1"].correct is None
    assert response.played == 1 and response.correct == int(bool(done.correct))
    assert world.service.fixture_predictions(days=400, client="").count == 4  # tous les clients


def test_awaiting_result(world: ServingWorld) -> None:
    """Un match prédit dont la date est passée mais dont le tournoi n'est pas encore validé."""
    batch = world.service.predict([_request(match_body(match_id="2099-2-1"))])
    world.service.record(batch, [_request(match_body(match_id="2099-2-1"))], "r", "scoring")
    world.flush()
    later = world.frame.copy().iloc[-5:]
    later["match_id"] = [f"later-{i}" for i in range(len(later))]
    later["match_date"] = world.last_day + timedelta(days=5)
    world.add_matches(later)
    world.service.features.refresh()
    (fixture,) = world.service.fixture_predictions(days=30).fixtures
    assert fixture.status == "awaiting_result"


def test_existing_predictions_table_gains_the_new_column(tmp_path: Path) -> None:
    """Une base de la phase 7 (sans `tourney_name`) est mise à niveau au démarrage, sans perdre de ligne."""
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'old.sqlite'}")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE predictions (prediction_id VARCHAR(32) PRIMARY KEY, match_id VARCHAR(64))"))
        conn.execute(sa.text("INSERT INTO predictions VALUES ('a', 'm')"))
    ApiStore(engine).create_schema()
    ApiStore(engine).create_schema()  # idempotent
    with engine.connect() as conn:
        columns = {c["name"] for c in sa.inspect(conn).get_columns("predictions")}
        assert "tourney_name" in columns and conn.execute(sa.text("SELECT COUNT(*) FROM predictions")).scalar() == 1


def _request(body: dict[str, object]) -> MatchRequest:
    return MatchRequest.model_validate(body)


def _predicted(
    world: ServingWorld, body: dict[str, object], client: str = "scoring"
) -> tuple[PredictionBatch, list[MatchRequest], str, str]:
    request = _request(body)
    return world.service.predict([request]), [request], "req", client
