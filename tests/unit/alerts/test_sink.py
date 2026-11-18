"""Service d'alertes : enrichissement, filtre `teams_events`, dédoublonnage, historique, et son API HTTP."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from tennis_mlops.alerts.builders import CampaignLink, CardContext, FlowRunInfo, RegistryNotice
from tennis_mlops.alerts.sink import AlertService, PrefectHook, SilenceRequest, create_app
from tennis_mlops.alerts.teams import TeamsNotifier
from tennis_mlops.ingestion.store import IngestionStore
from tennis_mlops.orchestration.events import EventKind, PipelineEvent, PipelineEvents, Severity

RUN_ID = "5b0e8d1c-3f2a-4f7e-9a51-2c8f0e6d4b17"
WEBHOOK = SecretStr("https://example.invalid/workflows/abc?sig=SECRET")


class Reader:
    def __init__(self, *runs: FlowRunInfo) -> None:
        self.runs = {r.id: r for r in runs}

    def flow_run(self, flow_run_id: str) -> FlowRunInfo | None:
        return self.runs.get(flow_run_id)


class Campaigns:
    def for_flow_run(self, flow_run_id: str) -> CampaignLink | None:
        return CampaignLink("c0ffee00c0ffee00", "4", "FAILED")


class World:
    def __init__(self, tmp_path: Path, *runs: FlowRunInfo, teams_events: tuple[str, ...]) -> None:
        self.posted: list[dict[str, Any]] = []
        self.store = IngestionStore.from_url(f"sqlite:///{tmp_path / 'tennis.sqlite'}")
        self.store.create_schema()
        self.events = PipelineEvents(self.store.engine)

        def teams(request: httpx.Request) -> httpx.Response:
            self.posted.append(json.loads(request.content))
            return httpx.Response(202)

        self.service = AlertService(
            events=self.events,
            reader=Reader(*runs),
            notifier=TeamsNotifier(WEBHOOK, client=httpx.Client(transport=httpx.MockTransport(teams))),
            context=lambda: CardContext(
                "mac.local", "http://127.0.0.1:4201", "http://127.0.0.1:7282", datetime(2026, 11, 12, 10, tzinfo=UTC)
            ),
            teams_events=teams_events,
            store=self.store,
            campaigns=Campaigns(),
        )

    def titles(self) -> list[str]:
        return [m["attachments"][0]["content"]["body"][0]["items"][0]["text"] for m in self.posted]


def failed(flow: str = "entrainement-atp", state: str = "Failed") -> FlowRunInfo:
    return FlowRunInfo(id=RUN_ID, name="olive-falcon", flow_name=flow, state_type=state.upper(), state_name=state,
                       message="ValueError : trop peu de matchs", failed_task="TemporalSplitter.split")  # fmt: skip


def hook(event_id: str = "ev-1", state: str = "Failed") -> PrefectHook:
    return PrefectHook(event_id=event_id, event=f"prefect.flow-run.{state}", flow_run_id=f"prefect.flow-run.{RUN_ID}")


def test_a_failure_becomes_one_card_enriched_from_prefect_mlflow_and_the_database(tmp_path: Path) -> None:
    world = World(tmp_path, failed(), teams_events=("echec",))
    world.events.record(PipelineEvent(EventKind.MODEL_PROMOTED, Severity.SUCCESS, "v9 en production",
                                      model_version="9", payload={"production_after": "9"}))  # fmt: skip
    outcome = world.service.on_prefect(hook())
    assert (outcome.status, outcome.kind, outcome.title) == ("sent", "echec", "❌ Échec de l'entraînement")
    assert world.titles() == ["❌ Échec de l'entraînement"]
    card = json.dumps(world.posted[0], ensure_ascii=False)
    assert "v9 inchangée" in card and "TemporalSplitter.split" in card and "c0ff…e00 (FAILED)" in card
    assert f"http://127.0.0.1:4201/runs/flow-run/{RUN_ID}" in card

    again = world.service.on_prefect(hook())  # Prefect renvoie le même événement
    assert again.status == "duplicate" and len(world.posted) == 1
    (alert,) = world.service.recent()
    assert alert["delivery"] == "sent" and alert["flow_run_id"] == RUN_ID and alert["kind"] == "echec"


def test_quarantine_and_degraded_runs_get_their_own_cards(tmp_path: Path) -> None:
    quarantined = failed("ingestion-atp")
    world = World(tmp_path, quarantined, teams_events=("quarantaine", "degrade"))
    lot = {"lot_id": "2019-D001", "rows_received": 4, "failed_checks": []}
    world.events.record(
        PipelineEvent(
            EventKind.LOT_QUARANTINED, Severity.FAILURE, "Lot en quarantaine", flow_run_id=RUN_ID, payload=lot
        )
    )
    assert world.service.on_prefect(hook()).kind == "quarantaine"

    world.service.reader = Reader(failed("ingestion-atp", "Dégradé"))
    world.store.set_watermark(date(2019, 3, 10))
    outcome = world.service.on_prefect(hook("ev-2", "Dégradé"))
    assert outcome.kind == "degrade" and outcome.status == "sent"
    assert world.titles() == ["🚫 Lot refusé par Great Expectations", "⚠️ Ingestion dégradée : source injoignable"]
    assert "10/03/2019" in json.dumps(world.posted[1], ensure_ascii=False)  # « ingéré jusqu'au » : le watermark


def test_types_outside_teams_events_are_kept_but_not_sent(tmp_path: Path) -> None:
    world = World(tmp_path, failed(), teams_events=("rollback",))
    assert world.service.on_prefect(hook()).status == "filtered"
    assert world.posted == [] and world.service.recent()[0]["delivery"] == "filtered"


def test_completed_runs_and_unknown_runs(tmp_path: Path) -> None:
    done = FlowRunInfo(id=RUN_ID, name="x", flow_name="ingestion-atp", state_type="COMPLETED", state_name="Completed")
    world = World(tmp_path, done, teams_events=("echec",))
    assert world.service.on_prefect(hook()).status == "ignored"
    world.service.reader = Reader()  # run introuvable : la carte part quand même, avec ce que l'on sait
    outcome = world.service.on_prefect(hook("ev-9", "Crashed"))
    assert outcome.status == "sent" and outcome.title == "💥 Arrêt brutal de « flow inconnu »"


def test_registry_decisions(tmp_path: Path) -> None:
    world = World(tmp_path, teams_events=("rollback",))
    rollback = RegistryNotice(kind="rollback", model="atp-winner", version="9", production_before="11",
                              production_after="9", reason="plaintes", at="2026-11-12T10:00:00")  # fmt: skip
    assert world.service.on_registry(rollback).status == "sent"
    assert world.service.on_registry(rollback).status == "duplicate"
    promotion = rollback.model_copy(update={"kind": "promotion", "version": "12", "at": "2026-11-12T11:00:00"})
    assert world.service.on_registry(promotion).status == "filtered"  # promotion : désactivée par défaut
    assert world.service.on_registry(rollback.model_copy(update={"kind": "rejet"})).status == "ignored"
    assert world.titles() == ["↩️ Rollback : production v11 → v9"]
    assert [a["kind"] for a in world.service.recent()] == ["promotion", "rollback"]


def test_http_api(tmp_path: Path) -> None:
    world = World(tmp_path, failed(), teams_events=("echec",))
    client = TestClient(create_app(world.service))
    silence = {"active": False, "until": None, "reason": ""}
    assert client.get("/health").json() == {
        "status": "ok",
        "teams": True,
        "teams_events": ["echec"],
        "silence": silence,
    }
    raw = '{"event_id": "ev-1", "event": "prefect.flow-run.Failed", "flow_run_id": "prefect.flow-run.' + RUN_ID + '"}'
    response = client.post("/prefect", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 200 and response.json()["status"] == "sent"
    notice = {"kind": "rejet", "model": "atp-winner", "version": "3"}
    assert client.post("/registry", json=notice).json()["status"] == "ignored"
    assert [a["title"] for a in client.get("/alerts", params={"limit": 5}).json()] == ["❌ Échec de l'entraînement"]


def test_a_broken_database_never_stops_a_card(tmp_path: Path) -> None:
    world = World(tmp_path, failed(), teams_events=("echec",))
    world.service.events = PipelineEvents.from_url(f"sqlite:///{tmp_path / 'absent' / 'x.sqlite'}")
    world.service.store = None
    assert world.service.on_prefect(hook()).status == "sent"
    assert "aucune" in json.dumps(world.posted[0], ensure_ascii=False)  # production inconnue : dit tel quel


def test_a_persisting_incident_is_not_resent_every_two_minutes(tmp_path: Path) -> None:
    world = World(tmp_path, failed("ingestion-atp"), teams_events=("quarantaine", "echec"))
    lot = {"lot_id": "2015-580", "rows_received": 127, "failed_checks": []}
    world.events.record(
        PipelineEvent(
            EventKind.LOT_QUARANTINED, Severity.FAILURE, "Lot en quarantaine", flow_run_id=RUN_ID, payload=lot
        )
    )
    assert world.service.on_prefect(hook("ev-1")).status == "sent"
    repeated = world.service.on_prefect(hook("ev-2"))  # le pas suivant rejoue le même lot, refusé à nouveau
    assert repeated.status == "repeated" and "déjà signalé à" in repeated.detail
    assert len(world.posted) == 1 and [a["delivery"] for a in world.service.recent()] == ["repeated", "sent"]

    world.service.cooldown = timedelta(0)  # alert_cooldown_min: 0 → toujours envoyer
    assert world.service.on_prefect(hook("ev-3")).status == "sent"


def test_silence_keeps_the_cards_without_sending_them_then_ends_by_itself(tmp_path: Path) -> None:
    world = World(tmp_path, failed(), teams_events=("echec", "rollback"))
    now = [datetime(2026, 11, 18, 10, tzinfo=UTC)]
    world.service.clock = lambda: now[0]
    state = world.service.silence(SilenceRequest(minutes=30, reason="make scenarios"))
    assert state.active and state.until == datetime(2026, 11, 18, 10, 30, tzinfo=UTC)

    outcome = world.service.on_prefect(hook("ev-1"))
    assert outcome.status == "silenced" and "make scenarios" in outcome.detail and "11:30" in outcome.detail
    notice = RegistryNotice(kind="rollback", model="atp-winner", version="1", production_before="2",
                            production_after="1", reason="S7", at="2026-11-18T10:05:00")  # fmt: skip
    assert world.service.on_registry(notice).status == "silenced"
    assert world.posted == [] and [a["delivery"] for a in world.service.recent()] == ["silenced", "silenced"]

    now[0] += timedelta(minutes=31)  # échu : levé de lui-même
    assert not world.service.silence_state().active
    assert world.service.on_prefect(hook("ev-2")).status == "sent"  # un silence ne compte pas comme un envoi


def test_silence_over_http(tmp_path: Path) -> None:
    world = World(tmp_path, failed(), teams_events=("echec",))
    client = TestClient(create_app(world.service))
    body = client.post("/silence", json={"minutes": 20, "reason": "make scenarios"}).json()
    assert body["active"] and body["reason"] == "make scenarios"
    assert client.get("/health").json()["silence"]["active"]
    assert client.post("/silence", json={"minutes": 0}).status_code == 422
    assert client.post("/silence", json={"minutes": 10_000}).status_code == 422
    assert client.delete("/silence").json() == {"active": False, "until": None, "reason": ""}
    response = client.post("/prefect", json={"event_id": "ev-9", "event": "prefect.flow-run.Failed",
                                             "flow_run_id": RUN_ID})  # fmt: skip
    assert response.json()["status"] == "sent"
