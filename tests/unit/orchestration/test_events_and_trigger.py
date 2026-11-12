from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from tennis_mlops.ingestion.store import IngestionStore
from tennis_mlops.orchestration.events import EventKind, PipelineEvent, PipelineEvents, Severity
from tennis_mlops.orchestration.trigger import TRAINING_REQUESTED, TrainingTrigger


def test_record_read_and_deduplicate(tmp_path: Path) -> None:
    events = PipelineEvents.from_url(f"sqlite:///{tmp_path / 'e.sqlite'}")
    start = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    events.record(PipelineEvent(EventKind.INGESTION_COMPLETED, Severity.INFO, "Ingestion : 2 lots"), at=start)
    events.record(
        PipelineEvent(EventKind.ALERT, Severity.FAILURE, "❌ Échec", "carte envoyée", date(2016, 5, 1), "run-1",
                      ref="evt-1", payload={"delivery": "sent"}),
        at=start + timedelta(minutes=1),
    )  # fmt: skip
    recent = events.recent()
    assert [e["kind"] for e in recent] == ["alert", "ingestion_completed"]
    assert recent[0]["payload"] == {"delivery": "sent"} and recent[0]["flow_run_id"] == "run-1"
    assert events.seen("evt-1") and not events.seen("evt-2")
    assert events.last(EventKind.TRAINING_REQUESTED) is None
    assert [e["title"] for e in events.recent(kinds=(EventKind.INGESTION_COMPLETED,))] == ["Ingestion : 2 lots"]


def test_a_broken_database_never_breaks_the_caller(tmp_path: Path) -> None:
    broken = PipelineEvents(sa.create_engine(f"sqlite:///{tmp_path / 'absent' / 'x.sqlite'}"))
    assert broken.record(PipelineEvent(EventKind.ROLLBACK, Severity.WARNING, "rollback")) is False


class Store:
    """Compte de matchs fictif : `count_matches_ingested_after` est tout ce dont le déclencheur a besoin."""

    def __init__(self) -> None:
        self.total = 0
        self.calls: list[datetime | None] = []

    def count_matches_ingested_after(self, since: datetime | None) -> int:
        self.calls.append(since)
        return self.total


def test_training_is_requested_once_per_threshold(tmp_path: Path) -> None:
    events = PipelineEvents.from_url(f"sqlite:///{tmp_path / 'e.sqlite'}")
    store, emitted = Store(), []

    def emit(**event: Any) -> None:
        emitted.append(event)

    trigger = TrainingTrigger(store, events, threshold=100, emit=emit)  # type: ignore[arg-type]
    store.total = 60
    assert not trigger.check("run-1", date(2016, 1, 10)).requested and emitted == []
    store.total = 140
    decision = trigger.check("run-2", date(2016, 1, 17))
    assert (
        decision.requested
        and decision.summary() == "réentraînement demandé : 140 nouveau(x) match(s) validé(s), seuil 100"
    )
    assert emitted[0]["event"] == TRAINING_REQUESTED and emitted[0]["payload"]["simulated_date"] == "2016-01-17"
    requested = events.last(EventKind.TRAINING_REQUESTED)
    assert requested is not None and requested["flow_run_id"] == "run-2"
    store.total = 30  # compté depuis la demande précédente
    assert not trigger.check("run-3", None).requested
    assert store.calls[0] is None and store.calls[-1] is not None

    off = TrainingTrigger(store, events, threshold=0, emit=emit)  # type: ignore[arg-type]
    store.total = 10_000
    assert not off.check("run-4", None).requested
    assert "désactivé" in off.check("run-4", None).summary()


def test_matches_counted_after_an_instant(tmp_path: Path) -> None:
    store = IngestionStore.from_url(f"sqlite:///{tmp_path / 't.sqlite'}")
    store.create_schema()
    assert store.count_matches_ingested_after(None) == 0
    assert store.count_matches_ingested_after(datetime(2026, 1, 1, tzinfo=UTC)) == 0
