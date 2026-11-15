"""Briques du service : profils des joueurs, historique qui survit à une panne, écrivain, résultats à remonter."""

import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from tennis_mlops.serving.features import FeatureStore, player_profiles
from tennis_mlops.serving.persistence import ApiStore, AsyncWriter
from tests.support.serving import ServingWorld, match_body, serving_world
from tests.support.synthetic import synthetic_matches


def test_player_profiles_take_the_latest_played_match() -> None:
    frame = synthetic_matches(weeks=3, matches_per_week=10, players=6)
    profiles = player_profiles(frame)
    some = next(iter(profiles.values()))
    played = frame[(frame["winner_id"] == some.player_id) | (frame["loser_id"] == some.player_id)]
    assert some.matches == len(played[played["status"].isin({"completed", "retired"})])
    assert some.last_match == pd.Timestamp(played[played["status"] == "completed"]["match_date"].max()).date()
    older = some.snapshot(date(some.last_match.year + 2, some.last_match.month, some.last_match.day))
    assert older.age == pytest.approx((some.age or 0) + 2, abs=0.01)  # l'âge vieillit, le classement non
    assert older.rank == some.rank
    assert player_profiles(frame.iloc[0:0]) == {}


class FlakySource:
    def __init__(self, world: ServingWorld) -> None:
        self.inner, self.down = world.ingestion, False

    def fingerprint(self) -> str:
        if self.down:
            msg = "SELECT"
            raise OperationalError(msg, {}, Exception("base arrêtée"))
        return self.inner.fingerprint()

    def matches_frame(self) -> pd.DataFrame:
        return self.inner.matches_frame()

    def lots_summary(self) -> dict[str, Any]:
        return {"watermark": "2011-03-01"}


def test_history_keeps_serving_when_the_database_stops(tmp_path: Path) -> None:
    world = serving_world(tmp_path)
    source = FlakySource(world)
    store = FeatureStore(source, world.service.features.config)
    first = store.refresh().snapshot
    assert first is not None and first.watermark == date(2011, 3, 1)
    assert first.default_match_date() == max(date(2011, 3, 1), first.last_day + pd.Timedelta(days=1))  # type: ignore[operator]
    source.down = True
    failed = store.refresh(force=True)
    assert not failed.changed and failed.snapshot is first and "base arrêtée" in (failed.error or "")
    assert store.current is first and store.last_error


def test_writer_counts_what_it_cannot_keep(tmp_path: Path) -> None:
    store = ApiStore.from_url(f"sqlite:///{tmp_path / 'a.sqlite'}")
    store.create_schema()
    writer = AsyncWriter(store, max_queue=100, batch=2)
    row = {"received_at": datetime.now(UTC), "method": "GET", "path": "/x", "status_code": 200, "duration_ms": 1.0,
           "predictions": 0}  # fmt: skip
    for i in range(101):
        writer.put("api_requests", {**row, "request_id": f"r{i}"})
    assert writer.dropped == 1 and writer.pending == 100
    assert writer.flush() == 100 and writer.pending == 0  # par paquets de 2, jusqu'au bout
    writer.put("api_requests", {**row, "request_id": "r0"})  # doublon : refusé par la base, compté, pas levé
    writer.flush()
    assert writer.failed == 1 and writer.written == 100


def test_background_thread_and_writer(tmp_path: Path) -> None:
    world = serving_world(tmp_path)
    world.app.router.lifespan_context = lifespan_of(world)  # comme en production : le service vit avec l'application
    with TestClient(world.app) as client:  # démarre le service : tables, écrivain, rafraîchissement
        deadline = time.monotonic() + 30
        while client.get("/ready").json()["status"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.2)
        assert client.post("/predict", json=match_body(match_id="m-9")).status_code == 200
        deadline = time.monotonic() + 10
        while not world.rows("SELECT * FROM predictions") and time.monotonic() < deadline:
            time.sleep(0.2)
    assert world.rows("SELECT match_id FROM predictions") == [{"match_id": "m-9"}]


def lifespan_of(world: ServingWorld) -> Any:
    from tennis_mlops.api.app import create_app

    return create_app(world.service).router.lifespan_context


def test_results_awaiting_feedback(tmp_path: Path) -> None:
    world = serving_world(tmp_path).ready()
    stored = world.frame[world.frame["status"] == "completed"].iloc[-1]
    body = match_body(int(stored["winner_id"]), int(stored["loser_id"]), match_id=str(stored["match_id"]))
    body["match_date"] = None  # le match est déjà en base dans ce test : on prédit « pour demain » avec son id
    world.client.post("/predict", json={k: v for k, v in body.items() if v is not None})
    world.client.post("/predict", json=match_body(match_id="pas-encore-joue"))
    world.flush()
    assert world.store.results_awaiting_feedback() == [(str(stored["match_id"]), int(stored["winner_id"]))]
    world.client.post("/feedback", json={"match_id": str(stored["match_id"]), "winner_id": int(stored["winner_id"])})
    assert world.store.results_awaiting_feedback() == []
    (metrics,) = world.store.online_metrics()
    assert (metrics.model_version, metrics.predictions, metrics.with_feedback) == ("1", 2, 1)
