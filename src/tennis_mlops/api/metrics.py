"""Métriques Prometheus de l'API (`GET /metrics`), lues par le dashboard technique et croisé (phase 9).

Un registre par application (pas le registre global) : plusieurs applications de test cohabitent sans se mélanger.
Les libellés restent à faible cardinalité : le chemin est le gabarit de la route (`/predict`, pas une URL brute).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from tennis_mlops.serving.service import PredictionService

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0)
BATCH_BUCKETS = (1, 2, 5, 10, 25, 50, 100, 200, 500, 1000)


class ApiMetrics:
    """Compteurs et jauges ; les jauges d'état sont relues au moment du scrape."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("tennis_api_requests_total", "Requêtes HTTP", ["method", "path", "status"], registry=r)
        self.latency = Histogram("tennis_api_request_duration_seconds", "Durée des requêtes", ["path"],
                                 buckets=LATENCY_BUCKETS, registry=r)  # fmt: skip
        self.predictions = Counter("tennis_api_predictions_total", "Matchs prédits", ["model_version", "client"],
                                   registry=r)  # fmt: skip
        self.batch_size = Histogram("tennis_api_batch_size", "Matchs par appel de prédiction", buckets=BATCH_BUCKETS,
                                    registry=r)  # fmt: skip
        self.feedbacks = Counter("tennis_api_feedback_predictions_total", "Prédictions jugées par un vrai résultat",
                                 ["model_version", "correct"], registry=r)  # fmt: skip
        self.errors = Counter("tennis_api_errors_total", "Réponses en erreur", ["type"], registry=r)
        self.model_info = Gauge("tennis_api_model_info", "Version servie (1 = servie)", ["name", "version"], registry=r)
        self.model_stale = Gauge("tennis_api_model_stale", "1 si le registre n'a pas pu être relu", registry=r)
        self.model_reloads = Gauge("tennis_api_model_reloads", "Changements de version depuis le démarrage",
                                   registry=r)  # fmt: skip
        self.model_load = Gauge("tennis_api_model_load_seconds", "Durée du dernier chargement", registry=r)
        self.features_day = Gauge("tennis_api_features_last_day_timestamp", "Dernière journée de l'historique",
                                  registry=r)  # fmt: skip
        self.features_build = Gauge("tennis_api_features_build_seconds", "Durée du dernier rejeu", registry=r)
        self.features_players = Gauge("tennis_api_features_players", "Joueurs connus", registry=r)
        self.writer_pending = Gauge("tennis_api_writer_pending", "Traces en attente d'écriture", registry=r)
        self.writer_dropped = Gauge("tennis_api_writer_dropped", "Traces perdues (file pleine)", registry=r)
        self.writer_failed = Gauge("tennis_api_writer_failed", "Traces refusées par la base", registry=r)
        self.ready = Gauge("tennis_api_ready", "2 = prêt, 1 = dégradé, 0 = pas prêt", registry=r)
        self.drift = Gauge("tennis_api_feature_drift_psi", "PSI des entrées récentes contre la période d'entraînement",
                           ["feature"], registry=r)  # fmt: skip
        self.drift_sample = Gauge("tennis_api_drift_sample", "Prédictions récentes comparées", registry=r)
        self._served: tuple[str, str] | None = None

    def render(self, service: PredictionService) -> bytes:
        """Met à jour les jauges d'état puis produit le texte Prometheus."""
        status = service.model.status()
        served = status.served
        if self._served and (served is None or (served.name, served.version) != self._served):
            self.model_info.remove(*self._served)
            self._served = None
        if served is not None:
            self.model_info.labels(served.name, served.version).set(1)
            self._served = (served.name, served.version)
        self.model_stale.set(1 if status.stale else 0)
        self.model_reloads.set(service.model.reloads)
        self.model_load.set(status.load_seconds or 0)
        snapshot = service.features.current
        if snapshot is not None:
            if snapshot.last_day is not None:
                self.features_day.set(_timestamp(snapshot.last_day.isoformat()))
            self.features_build.set(snapshot.build_seconds)
            self.features_players.set(len(snapshot.profiles))
        self.writer_pending.set(service.writer.pending)
        self.writer_dropped.set(service.writer.dropped)
        self.writer_failed.set(service.writer.failed)
        self.ready.set({"ready": 2, "degraded": 1}.get(service.readiness().status, 0))
        drift = service.drift_report()
        for feature, value in drift.psi.items():
            self.drift.labels(feature).set(value)
        self.drift_sample.set(drift.sample)
        return generate_latest(self.registry)


def _timestamp(day: str) -> float:
    from datetime import UTC, datetime

    return datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp()
