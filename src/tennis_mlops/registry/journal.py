"""Journal des décisions : chaque promotion, rejet ou rollback devient un run de l'expérience `<modèle>-decisions`.

Le registre MLflow garde l'état (alias, tags) ; ce journal garde l'**historique** : qui a déplacé quoi, quand et
pourquoi, dans l'ordre, consultable dans l'UI MLflow (liste des runs, filtres sur `tags.event`).
"""

from __future__ import annotations

from mlflow.exceptions import MlflowException

from tennis_mlops.registry.events import RegistryEvent
from tennis_mlops.training.tracking import MlflowTracker


def journal_experiment(model: str) -> str:
    """Nom de l'expérience du journal."""
    return f"{model}-decisions"


class DecisionJournal:
    """Écrit un run par décision ; le run est fermé aussitôt."""

    def __init__(self, tracker: MlflowTracker) -> None:
        self.tracker = tracker

    @classmethod
    def for_model(cls, tracking_uri: str, model: str) -> DecisionJournal:
        """Journal du modèle sur le serveur MLflow donné."""
        return cls(MlflowTracker(tracking_uri, journal_experiment(model), check_server=False))

    def record(self, event: RegistryEvent) -> str:
        """Run du journal pour cet événement ; renvoie son id."""
        t = self.tracker
        tags = {
            "kind": "decision",
            "event": event.kind.value,
            "model": event.model,
            "version": event.version,
            "production_before": event.production_before or "",
            "production_after": event.production_after or "",
            "decided_by": event.decided_by.value,
            "reason": event.reason[:5000],
            "decided_at": event.at,
            "campaign_run_id": event.campaign_run_id or "",
        }
        if event.decision is not None:
            tags["gate_status"] = event.decision.status.value
        run_id = t.start_run(event.title, tags)
        if event.decision is not None:
            metrics = {"gate/passed": float(event.decision.passed),
                       "gate/rules_failed": float(len(event.decision.failed_rules))}  # fmt: skip
            if event.decision.delta_log_loss is not None:
                metrics["gate/delta_log_loss"] = event.decision.delta_log_loss
            t.log_metrics(run_id, metrics)
            t.log_frame(run_id, event.decision.table(), "decision/rules.csv")
        t.log_dict(run_id, event.as_dict(), "decision/event.json")
        t.finish(run_id)
        return run_id

    def try_record(self, event: RegistryEvent) -> str | None:
        """Comme `record`, sans jamais faire échouer l'appelant ; renvoie l'erreur éventuelle.

        La décision est déjà appliquée dans le registre (la source de vérité) : un journal incomplet ne doit pas
        faire croire qu'elle a échoué.
        """
        try:
            self.record(event)
        except (MlflowException, OSError) as exc:
            return f"journal des décisions non écrit : {exc}"
        return None
