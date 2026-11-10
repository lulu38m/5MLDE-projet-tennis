"""Une décision du registre : promotion, rejet, promotion manuelle ou rollback, datée et motivée."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tennis_mlops.registry.gate import GateDecision
from tennis_mlops.registry.tagging import DecidedBy


class EventKind(StrEnum):
    """Les quatre décisions possibles."""

    PROMOTION = "promotion"
    REJECTION = "rejet"
    MANUAL_PROMOTION = "promotion manuelle"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class RegistryEvent:
    """Ce qui a changé dans le registre, par qui, quand et pourquoi."""

    kind: EventKind
    model: str
    version: str  # la version promue, rejetée ou restaurée
    production_before: str | None
    production_after: str | None
    decided_by: DecidedBy
    reason: str
    at: str
    decision: GateDecision | None = None
    campaign_run_id: str | None = None

    @property
    def title(self) -> str:
        """Nom court (nom du run dans le journal MLflow)."""
        if self.kind is EventKind.ROLLBACK:
            return f"rollback v{self.production_before} → v{self.production_after}"
        return f"{self.kind.value} v{self.version}"

    def summary(self) -> str:
        """Une phrase pour le terminal et les logs."""
        before = f"v{self.production_before}" if self.production_before else "aucune"
        if self.kind is EventKind.REJECTION:
            return f"{self.model} v{self.version} refusée — production inchangée ({before}) : {self.reason}"
        if self.kind is EventKind.ROLLBACK:
            return f"rollback : production {before} → v{self.production_after} ({self.reason})"
        return f"{self.model} v{self.version} en production (avant : {before}) — {self.reason}"

    def as_dict(self) -> dict[str, Any]:
        """Version JSON (artefact du journal)."""
        return {
            "kind": self.kind.value,
            "model": self.model,
            "version": self.version,
            "production_before": self.production_before,
            "production_after": self.production_after,
            "decided_by": self.decided_by.value,
            "reason": self.reason,
            "at": self.at,
            "campaign_run_id": self.campaign_run_id,
            "gate": self.decision.as_dict() if self.decision else None,
        }
