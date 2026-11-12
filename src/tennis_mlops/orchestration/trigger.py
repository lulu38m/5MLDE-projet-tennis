"""Déclenchement de l'entraînement par événement : assez de nouveaux matchs validés → événement Prefect.

L'ingestion ne lance pas l'entraînement elle-même : elle compte les matchs stockés depuis la dernière demande et,
au-delà du seuil (`orchestration.training_trigger_matches`), elle émet l'événement `tennis.training.requested`.
Une automation Prefect y réagit en lançant le déploiement `entrainement-atp` (docs/ORCHESTRATION.md). Les deux
flows restent indépendants : on peut entraîner à la main, et une ingestion sans automation n'entraîne rien.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from prefect.events import emit_event

from tennis_mlops.ingestion.store import IngestionStore
from tennis_mlops.orchestration.events import EventKind, PipelineEvent, PipelineEvents, Severity

TRAINING_REQUESTED = "tennis.training.requested"
DATASET_RESOURCE = "tennis.dataset.atp"

Emitter = Callable[..., Any]


@dataclass(frozen=True)
class TriggerDecision:
    """Ce qu'a décidé le déclencheur après une ingestion."""

    requested: bool
    new_matches: int
    threshold: int
    since: datetime | None = None

    def summary(self) -> str:
        """Une ligne pour les logs."""
        if self.threshold == 0:
            return "déclenchement de l'entraînement désactivé (training_trigger_matches = 0)"
        verdict = "réentraînement demandé" if self.requested else "pas de réentraînement"
        return f"{verdict} : {self.new_matches} nouveau(x) match(s) validé(s), seuil {self.threshold}"


class TrainingTrigger:
    """Compte les nouveaux matchs et émet l'événement de demande d'entraînement."""

    def __init__(
        self, store: IngestionStore, events: PipelineEvents, threshold: int, emit: Emitter = emit_event
    ) -> None:
        self.store = store
        self.events = events
        self.threshold = threshold
        self.emit = emit

    def check(self, flow_run_id: str | None, simulated_date: date | None) -> TriggerDecision:
        """Émet `tennis.training.requested` si le seuil est atteint depuis la dernière demande."""
        last = self.events.last(EventKind.TRAINING_REQUESTED)
        since = last["occurred_at"] if last else None
        new = self.store.count_matches_ingested_after(since)
        decision = TriggerDecision(self.threshold > 0 and new >= self.threshold, new, self.threshold, since)
        if not decision.requested:
            return decision
        payload = {
            "new_matches": new,
            "threshold": self.threshold,
            "flow_run_id": flow_run_id,
            "simulated_date": simulated_date.isoformat() if simulated_date else None,
        }
        self.events.record(
            PipelineEvent(
                EventKind.TRAINING_REQUESTED,
                Severity.INFO,
                f"Réentraînement demandé : {new} nouveaux matchs",
                f"seuil {self.threshold}",
                simulated_date,
                flow_run_id,
                payload=payload,
            )
        )
        self.emit(
            event=TRAINING_REQUESTED,
            resource={"prefect.resource.id": DATASET_RESOURCE, "prefect.resource.name": "matchs ATP validés"},
            payload=payload,
        )
        return decision
