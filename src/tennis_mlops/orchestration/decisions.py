"""Annoncer une décision du registre : une ligne dans `pipeline_events`, et le service d'alertes prévenu.

Utilisé par le flow d'entraînement (promotion ou rejet en fin de campagne) et par `make promote` / `make rollback`.
Les deux sont facultatifs et jamais bloquants : la décision est déjà appliquée dans MLflow quand on l'annonce.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from tennis_mlops.alerts.builders import RegistryNotice
from tennis_mlops.alerts.client import AlertSinkClient
from tennis_mlops.orchestration.events import EventKind, PipelineEvent, PipelineEvents, Severity
from tennis_mlops.registry.events import EventKind as Decision
from tennis_mlops.registry.events import RegistryEvent

KINDS = {
    Decision.PROMOTION: (EventKind.MODEL_PROMOTED, Severity.SUCCESS),
    Decision.REJECTION: (EventKind.MODEL_REJECTED, Severity.INFO),
    Decision.MANUAL_PROMOTION: (EventKind.MANUAL_PROMOTION, Severity.WARNING),
    Decision.ROLLBACK: (EventKind.ROLLBACK, Severity.WARNING),
}


def decision_title(event: RegistryEvent) -> str:
    """Titre court (annotation Grafana)."""
    if event.kind is Decision.ROLLBACK:
        return f"Rollback {event.model} : v{event.production_before} → v{event.production_after}"
    if event.kind is Decision.REJECTION:
        return f"{event.model} v{event.version} refusée par la règle de mise en service"
    manual = " (décision manuelle)" if event.kind is Decision.MANUAL_PROMOTION else ""
    return f"{event.model} v{event.version} en production{manual}"


def decision_event(
    event: RegistryEvent, flow_run_id: str | None = None, simulated_date: date | None = None
) -> PipelineEvent:
    """La ligne de `pipeline_events` d'une décision ; `payload.production_after` dit la version servie ensuite."""
    kind, severity = KINDS[event.kind]
    return PipelineEvent(
        kind,
        severity,
        decision_title(event),
        event.reason,
        simulated_date,
        flow_run_id,
        model_version=event.version,
        payload=event.as_dict(),
    )


@dataclass
class DecisionAnnouncer:
    """Journal et alerte d'une décision ; chaque côté est facultatif."""

    events: PipelineEvents | None = None
    sink: AlertSinkClient | None = None

    def announce(
        self, event: RegistryEvent, metrics: Mapping[str, Any] | None = None, flow_run_id: str | None = None
    ) -> list[str]:
        """Les lignes à afficher (ce qui n'a pas pu être fait, et ce qu'est devenue l'alerte)."""
        lines: list[str] = []
        if self.events is not None and not self.events.record(decision_event(event, flow_run_id)):
            lines.append("attention : décision non inscrite dans pipeline_events (base injoignable)")
        if self.sink is not None:
            notice = RegistryNotice.model_validate({**event.as_dict(), "metrics": dict(metrics or {})})
            lines.append(self.sink.notify_registry(notice).line())
        return lines
