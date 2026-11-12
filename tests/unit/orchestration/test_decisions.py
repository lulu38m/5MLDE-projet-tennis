"""Annonce d'une décision du registre : ligne dans `pipeline_events`, service d'alertes prévenu, jamais bloquant."""

import json
from pathlib import Path
from typing import Any

import httpx

from tennis_mlops.alerts.client import AlertSinkClient
from tennis_mlops.orchestration.decisions import DecisionAnnouncer, decision_title
from tennis_mlops.orchestration.events import EventKind, PipelineEvents
from tennis_mlops.registry.events import EventKind as Decision
from tennis_mlops.registry.events import RegistryEvent
from tennis_mlops.registry.tagging import DecidedBy


def decision(kind: Decision, before: str | None = "9", after: str | None = "11", version: str = "11") -> RegistryEvent:
    return RegistryEvent(kind, "atp-winner", version, before, after, DecidedBy.PIPELINE, "toutes les règles passent",
                         "2026-11-12T10:00:00+00:00", campaign_run_id="camp")  # fmt: skip


def sink(status: int, seen: list[dict[str, Any]], reply: str = "sent") -> AlertSinkClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(status, json={"status": reply, "detail": "HTTP 202"})

    return AlertSinkClient("http://alert-sink", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_promotion_is_journaled_and_sent_with_its_figures(tmp_path: Path) -> None:
    events = PipelineEvents.from_url(f"sqlite:///{tmp_path / 'e.sqlite'}")
    seen: list[dict[str, Any]] = []
    lines = DecisionAnnouncer(events, sink(200, seen)).announce(decision(Decision.PROMOTION), {"log_loss": 0.6}, "run")
    assert lines == ["alerte : carte envoyée sur Teams"]
    assert seen[0]["kind"] == "promotion" and seen[0]["metrics"] == {"log_loss": 0.6}
    row = events.last(EventKind.MODEL_PROMOTED)
    assert row is not None and row["title"] == "atp-winner v11 en production" and row["flow_run_id"] == "run"
    assert row["payload"]["production_after"] == "11" and row["severity"] == "success"


def test_titles_of_each_decision() -> None:
    assert decision_title(decision(Decision.REJECTION, after="9")) == (
        "atp-winner v11 refusée par la règle de mise en service"
    )
    assert decision_title(decision(Decision.ROLLBACK, "11", "9", "9")) == "Rollback atp-winner : v11 → v9"
    assert decision_title(decision(Decision.MANUAL_PROMOTION)).endswith("(décision manuelle)")


def test_nothing_reachable_is_reported_not_raised(tmp_path: Path) -> None:
    events = PipelineEvents.from_url(f"sqlite:///{tmp_path / 'absent' / 'e.sqlite'}")
    unreachable = AlertSinkClient("http://127.0.0.1:9", timeout_s=0.5)
    lines = DecisionAnnouncer(events, unreachable).announce(decision(Decision.ROLLBACK, "11", "9", "9"))
    assert lines[0].startswith("attention : décision non inscrite")
    assert lines[1] == "alerte non transmise : service d'alertes injoignable (ConnectError)"
    seen: list[dict[str, Any]] = []
    assert DecisionAnnouncer(None, sink(500, seen)).announce(decision(Decision.PROMOTION)) == [
        "alerte non transmise : service d'alertes : HTTP 500"
    ]
    assert (
        DecisionAnnouncer(None, sink(200, seen, "failed"))
        .announce(decision(Decision.PROMOTION))[0]
        .startswith("alerte : envoi Teams en échec")
    )
    assert DecisionAnnouncer().announce(decision(Decision.PROMOTION)) == []
