"""Service d'alertes (`alert_sink`) : reçoit les événements, construit la carte, l'envoie sur Teams, s'en souvient.

```
automation Prefect (Failed, Crashed, Dégradé) ─► POST /prefect   ─┐
registre (rollback, promotion)                 ─► POST /registry ─┼─► carte ─► filtre teams_events ─► Teams
                                                                   └─► pipeline_events (kind = alert) : historique,
                                                                       dédoublonnage, annotations Grafana
```

- **Une alerte = un incident** : l'automation ne se déclenche que sur l'état final d'un flow (après ses retries) ;
  un événement déjà traité (même identifiant) est ignoré, même si Prefect le renvoie.
- **Enrichissement** : l'automation ne transmet que des identifiants ; le service relit le flow run dans Prefect,
  les lots refusés dans `pipeline_events`, la campagne dans MLflow, la version servie et les données disponibles.
- **Filtre** : seuls les types de `orchestration.teams_events` partent sur Teams ; les autres sont gardés dans
  l'historique (`make alerts`) avec le statut `filtered`. Sans webhook, rien ne part (`disabled`).
- **Pas de rafale** : la simulation repasse chaque minute ; un incident qui persiste (le même lot refusé, la
  source toujours injoignable) ne renvoie une carte qu'après `orchestration.alert_cooldown_min` (statut `repeated`
  entre-temps). Le canal Teams est partagé avec la classe.
- **Jamais bloquant** : une lecture qui échoue donne une carte moins riche, pas d'erreur ; l'URL du webhook n'est
  jamais écrite.

Lancement : service `alert_sink` de la stack (`python -m tennis_mlops.alerts serve`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from tennis_mlops.alerts.builders import (
    AlertKind,
    CampaignLink,
    CardContext,
    FlowRunInfo,
    PrefectEvent,
    RegistryNotice,
    Situation,
    classify,
    degraded_alert,
    failure_alert,
    promotion_alert,
    quarantine_alert,
    rollback_alert,
)
from tennis_mlops.alerts.cards import DISPLAY_ZONE, Alert, Tone, adaptive_card
from tennis_mlops.alerts.prefect_reader import FlowRunReader, flow_run_id_of
from tennis_mlops.alerts.teams import Delivery, TeamsNotifier
from tennis_mlops.ingestion.store import IngestionStore
from tennis_mlops.orchestration.events import EventKind, PipelineEvent, PipelineEvents, Severity
from tennis_mlops.validation.suites import SUITES_VERSION

logger = logging.getLogger(__name__)

FILTERED, DUPLICATE, IGNORED, REPEATED = "filtered", "duplicate", "ignored", "repeated"
SEVERITY = {Tone.FAILURE: Severity.FAILURE, Tone.WARNING: Severity.WARNING, Tone.SUCCESS: Severity.SUCCESS,
            Tone.INFO: Severity.INFO}  # fmt: skip
REGISTRY_KINDS = {"rollback": AlertKind.ROLLBACK, "promotion": AlertKind.PROMOTION,
                  "promotion manuelle": AlertKind.PROMOTION}  # fmt: skip
PRODUCTION_CHANGES = (EventKind.MODEL_PROMOTED, EventKind.MANUAL_PROMOTION, EventKind.ROLLBACK)


class PrefectHook(BaseModel):
    """Corps envoyé par l'automation (gabarit Jinja de `orchestration/setup.py`)."""

    event_id: str = ""
    event: str = ""  # prefect.flow-run.Failed…
    flow_run_id: str  # prefect.flow-run.<uuid> ou <uuid>


class Outcome(BaseModel):
    """Ce que le service a fait d'un événement."""

    status: str  # sent, disabled, failed, filtered, repeated, duplicate, ignored
    kind: str | None = None
    title: str | None = None
    detail: str = ""


class CampaignFinder(Protocol):
    """Retrouve la campagne MLflow d'un flow run d'entraînement."""

    def for_flow_run(self, flow_run_id: str) -> CampaignLink | None:
        """La campagne, ou None."""
        ...


@dataclass
class AlertService:
    """Le cœur du service, indépendant de HTTP (testé directement)."""

    events: PipelineEvents
    reader: FlowRunReader
    notifier: TeamsNotifier
    context: Callable[[], CardContext]
    teams_events: Collection[str]
    store: IngestionStore | None = None
    campaigns: CampaignFinder | None = None
    suite: str = SUITES_VERSION
    cooldown: timedelta = timedelta(minutes=30)  # même incident déjà signalé : pas de nouvelle carte avant ce délai

    # ------------------------------------------------------------------ entrées

    def on_prefect(self, hook: PrefectHook) -> Outcome:
        """Un flow s'est terminé en échec, en arrêt brutal ou dégradé."""
        run_id = flow_run_id_of(hook.flow_run_id)
        ref = (hook.event_id or f"{run_id}:{hook.event}")[:64]
        if self._seen(ref):
            return Outcome(status=DUPLICATE, detail=f"événement {ref} déjà traité")
        state_name = hook.event.rsplit(".", 1)[-1] or "Failed"
        info = self.reader.flow_run(run_id) or FlowRunInfo(
            id=run_id, name="?", flow_name="flow inconnu", state_type=state_name.upper(), state_name=state_name,
            message="flow run introuvable dans Prefect au moment de l'alerte",
        )  # fmt: skip
        if info.state_type.upper() == "COMPLETED" and info.state_name != "Dégradé":
            return Outcome(status=IGNORED, detail=f"{info.flow_name} terminé normalement ({info.state_name})")
        event = PrefectEvent(hook.event_id or "?", hook.event or f"prefect.flow-run.{info.state_name}")
        ctx = self.context()
        quarantined = self._quarantined(info.id)
        kind = classify(info, quarantined)
        if kind is AlertKind.DEGRADED:
            alert = degraded_alert(info, self.situation(), event, ctx)
            signature = f"{kind.value}:{info.flow_name}"
        elif kind is AlertKind.QUARANTINE:
            alert = quarantine_alert(info, quarantined, event, ctx, self.suite)
            signature = f"{kind.value}:" + ",".join(sorted(str(lot.get("lot_id")) for lot in quarantined))
        else:
            campaign = self.campaigns.for_flow_run(info.id) if self.campaigns else None
            alert = failure_alert(info, event, ctx, self.situation(), campaign)
            signature = f"{kind.value}:{info.flow_name}:{info.failed_task}:{info.message[:120]}"
        return self._deliver(kind, alert, ref, info.id, signature=signature)

    def on_registry(self, notice: RegistryNotice) -> Outcome:
        """Une décision du registre : rollback ou mise en production (un rejet n'est jamais notifié)."""
        kind = REGISTRY_KINDS.get(notice.kind)
        if kind is None:
            return Outcome(status=IGNORED, detail=f"« {notice.kind} » : décision normale, pas d'alerte")
        if self._seen(notice.ref):
            return Outcome(status=DUPLICATE, kind=kind.value, detail=f"décision {notice.ref} déjà notifiée")
        ctx = self.context()
        alert = rollback_alert(notice, ctx) if kind is AlertKind.ROLLBACK else promotion_alert(notice, ctx)
        return self._deliver(kind, alert, notice.ref, None, model_version=notice.version, signature=notice.ref)

    # ------------------------------------------------------------------ lectures (jamais bloquantes)

    def situation(self) -> Situation:
        """Version servie (dernier changement de production) et données disponibles."""
        production = None
        summary: dict[str, Any] = {}
        try:
            last = self.events.last(*PRODUCTION_CHANGES)
            production = (last or {}).get("payload", {}).get("production_after") or (last or {}).get("model_version")
            if self.store is not None:
                summary = self.store.lots_summary()
        except SQLAlchemyError as exc:
            logger.warning("situation incomplète : %s", exc)
        return Situation(
            production=str(production) if production else None,
            data_end=summary.get("last_valid_lot_end"),
            watermark=summary.get("watermark"),
            lots_valid=summary.get("lots_valid"),
        )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Les dernières alertes (sans le JSON des cartes)."""
        rows = self.events.recent(limit=limit, kinds=(EventKind.ALERT,))
        return [
            {
                "at": row["occurred_at"].isoformat() if row["occurred_at"] else None,
                "kind": row["payload"].get("kind"),
                "delivery": row["payload"].get("delivery"),
                "title": row["title"],
                "detail": row["payload"].get("detail", ""),
                "flow_run_id": row["flow_run_id"],
            }
            for row in rows
        ]

    def _seen(self, ref: str) -> bool:
        try:
            return self.events.seen(ref)
        except SQLAlchemyError as exc:
            logger.warning("dédoublonnage impossible (%s) : l'alerte est traitée", exc)
            return False

    def _quarantined(self, flow_run_id: str) -> list[dict[str, Any]]:
        try:
            rows = self.events.recent(limit=20, kinds=(EventKind.LOT_QUARANTINED,), flow_run_id=flow_run_id)
        except SQLAlchemyError as exc:
            logger.warning("lots refusés illisibles : %s", exc)
            return []
        return [row["payload"] for row in reversed(rows) if row["payload"]]

    # ------------------------------------------------------------------ envoi et historique

    def _already_sent(self, signature: str) -> datetime | None:
        """Quand ce même incident a été envoyé, s'il l'a été depuis moins de `cooldown`."""
        if self.cooldown <= timedelta(0):
            return None
        since = datetime.now(UTC) - self.cooldown
        try:
            rows = self.events.recent(limit=100, kinds=(EventKind.ALERT,))
        except SQLAlchemyError:
            return None
        for row in rows:
            at = row["occurred_at"] if row["occurred_at"].tzinfo else row["occurred_at"].replace(tzinfo=UTC)
            if at < since:
                break
            if row["payload"].get("signature") == signature and row["payload"].get("delivery") == Delivery.SENT:
                return cast("datetime", at)
        return None

    def _deliver(
        self,
        kind: AlertKind,
        alert: Alert,
        ref: str,
        flow_run_id: str | None,
        *,
        signature: str,
        model_version: str | None = None,
    ) -> Outcome:
        if kind.value not in self.teams_events:
            status, detail = FILTERED, f"« {kind.value} » absent de orchestration.teams_events"
        elif (sent_at := self._already_sent(signature)) is not None:
            when = sent_at.astimezone(DISPLAY_ZONE).strftime("%H:%M")
            status, detail = (
                REPEATED,
                f"même incident déjà signalé à {when} : pas de nouvelle carte avant {self.cooldown}",
            )
        else:
            result = self.notifier.send(alert)
            status, detail = result.status.value, result.detail
        level = logging.WARNING if status == Delivery.FAILED else logging.INFO
        logger.log(level, "alerte %s (%s) : %s — %s", kind.value, status, alert.title, detail)
        self.events.record(
            PipelineEvent(
                EventKind.ALERT,
                SEVERITY[alert.tone],
                alert.title,
                f"{status} · {detail}",
                flow_run_id=flow_run_id,
                model_version=model_version,
                ref=ref,
                payload={
                    "kind": kind.value,
                    "delivery": status,
                    "detail": detail,
                    "signature": signature,
                    "card": adaptive_card(alert),
                },
            )
        )
        return Outcome(status=status, kind=kind.value, title=alert.title, detail=detail)


class MlflowCampaigns:
    """Campagne MLflow d'un flow run (tag `prefect.flow_run_id` posé par le flow d'entraînement)."""

    def __init__(self, tracking_uri: str, experiment: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment

    def for_flow_run(self, flow_run_id: str) -> CampaignLink | None:
        """None si MLflow ne répond pas ou si la campagne n'a pas été ouverte (échec avant MLflow)."""
        from mlflow.tracking import MlflowClient

        try:
            client = MlflowClient(self.tracking_uri)
            experiment = client.get_experiment_by_name(self.experiment)
            if experiment is None:
                return None
            runs = client.search_runs(
                [experiment.experiment_id],
                filter_string=f"tags.`prefect.flow_run_id` = '{flow_run_id}' and tags.kind = 'campaign'",
                max_results=1,
            )
        except Exception as exc:
            logger.warning("campagne MLflow introuvable : %s", type(exc).__name__)
            return None
        if not runs:
            return None
        info = runs[0].info
        return CampaignLink(info.run_id, info.experiment_id, info.status)


def create_app(service: AlertService) -> FastAPI:
    """L'application HTTP autour du service."""
    app = FastAPI(title="tennis-mlops · alert_sink", version="1.0", description=__doc__)

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Le service répond ; Teams configuré ou non, et quels types partent."""
        return {"status": "ok", "teams": service.notifier.enabled, "teams_events": sorted(service.teams_events)}

    @app.post("/prefect")
    def prefect(hook: PrefectHook) -> Outcome:
        """Appelé par l'automation Prefect sur l'état final d'un flow."""
        return service.on_prefect(hook)

    @app.post("/registry")
    def registry(notice: RegistryNotice) -> Outcome:
        """Appelé par le flow d'entraînement et par `make rollback` / `make promote`."""
        return service.on_registry(notice)

    @app.get("/alerts")
    def alerts(limit: int = 20) -> list[dict[str, Any]]:
        """Historique des alertes (envoyées, filtrées, en échec)."""
        return service.recent(min(max(limit, 1), 200))

    return app


def card_context_factory(host: str, owner: str, prefect_ui: str, mlflow_ui: str) -> Callable[[], CardContext]:
    """Contexte des cartes, daté à chaque alerte."""

    def make() -> CardContext:
        return CardContext(host=host, prefect_ui=prefect_ui, mlflow_ui=mlflow_ui, at=datetime.now(UTC), owner=owner)

    return make
