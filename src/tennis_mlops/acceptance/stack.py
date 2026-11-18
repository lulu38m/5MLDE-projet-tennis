"""La stack vue par les tests d'acceptation : chaque service lu ou piloté par son interface publique.

Les scénarios ne touchent à rien en coulisses : ils passent par les mêmes portes qu'un utilisateur ou qu'un
opérateur (API d'administration de `mock_source`, API de prédiction, service d'alertes, registre MLflow, API
Prefect, base du projet en lecture). Les adresses viennent de `.env` (`Settings`).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import sqlalchemy as sa

from tennis_mlops.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, ProjectConfig, Settings
from tennis_mlops.ingestion.store import IngestionStore
from tennis_mlops.orchestration.clock import ClockStep, SimulationClock
from tennis_mlops.orchestration.events import PipelineEvents
from tennis_mlops.registry.model_registry import ModelRegistry
from tennis_mlops.registry.tagging import Alias
from tennis_mlops.training.config import DEFAULT_TRAIN_CONFIG_PATH, TrainConfig

HTTP_TIMEOUT_S = 15.0
CLIENT = "scenarios"  # X-Client des requêtes des scénarios : comptées à part dans Grafana
TRAINING_FLOW = "entrainement-atp"
OUR_FLOWS = ("simulation-atp", "ingestion-atp", TRAINING_FLOW, "scoring-atp", "feedback-atp")
FINAL = {"COMPLETED", "FAILED", "CRASHED", "CANCELLED"}


class WaitTimeoutError(TimeoutError):
    """Ce qu'on attendait n'est pas arrivé à temps."""


def wait_for[T](probe: Callable[[], T | None], timeout_s: float, what: str, interval_s: float = 2.0) -> T:
    """Rappelle `probe` jusqu'à ce qu'elle rende autre chose que None (ou False) ; sinon `WaitTimeoutError`.

    Une erreur réseau pendant l'attente compte comme « pas encore » : on attend justement qu'un service revienne.
    """
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while True:
        try:
            value = probe()
        except WaitTimeoutError:  # c'est un TimeoutError, donc un OSError : à ne pas confondre avec « pas encore »
            raise
        except (httpx.HTTPError, sa.exc.SQLAlchemyError, OSError) as exc:
            value, last_error = None, f" (dernière erreur : {type(exc).__name__})"
        if value is not None and value is not False:
            return value
        if time.monotonic() >= deadline:
            msg = f"{what} : pas au bout de {timeout_s:.0f} s{last_error}"
            raise WaitTimeoutError(msg)
        time.sleep(interval_s)


@dataclass(frozen=True)
class RunInfo:
    """Un flow run Prefect, ce que les scénarios en regardent."""

    id: str
    name: str
    flow_name: str
    state_type: str  # COMPLETED, FAILED…
    state_name: str  # Completed, Failed, Dégradé…
    created: datetime | None

    @property
    def final(self) -> bool:
        """Terminé (dans un sens ou dans l'autre)."""
        return self.state_type in FINAL


@dataclass(frozen=True)
class TaskInfo:
    """Une task run : son nom, son nombre d'essais, son état."""

    name: str
    run_count: int
    state_name: str
    seconds: float | None


@dataclass
class Stack:
    """Accès à la stack qui tourne (adresses de l'hôte)."""

    settings: Settings = field(default_factory=Settings)
    config: ProjectConfig = field(default_factory=lambda: ProjectConfig.from_yaml(DEFAULT_CONFIG_PATH))
    train: TrainConfig = field(default_factory=lambda: TrainConfig.from_yaml(DEFAULT_TRAIN_CONFIG_PATH))
    root: Path = PROJECT_ROOT
    http: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=HTTP_TIMEOUT_S))
    _engine: sa.Engine | None = field(default=None, init=False, repr=False)
    _registry: ModelRegistry | None = field(default=None, init=False, repr=False)

    def close(self) -> None:
        """Ferme les connexions."""
        self.http.close()
        if self._engine is not None:
            self._engine.dispose()

    # ------------------------------------------------------------------ adresses

    def url(self, service: str, path: str = "") -> str:
        """Adresse d'un service vu depuis l'hôte."""
        bases = {
            "mock_source": self.settings.mock_source_url, "api": self.settings.api_url,
            "alert_sink": self.settings.alert_sink_url, "mlflow": self.settings.mlflow_tracking_uri,
            "prefect": self.settings.prefect_ui_url, "ui": self.settings.ui_url,
            "grafana": self.settings.grafana_ui_url,
        }  # fmt: skip
        return f"{str(bases[service]).rstrip('/')}{path}"

    def prefect_run_url(self, flow_run_id: str) -> str:
        """Lien vers un flow run dans l'UI Prefect."""
        return f"{str(self.settings.prefect_ui_url).rstrip('/')}/runs/flow-run/{flow_run_id}"

    def mlflow_version_url(self, version: str) -> str:
        """Lien vers une version du registre dans l'UI MLflow."""
        return (
            f"{str(self.settings.mlflow_ui_url).rstrip('/')}/#/models/{self.train.registered_model}/versions/{version}"
        )

    # ------------------------------------------------------------------ santé

    def healthy(self, service: str) -> bool:
        """Le service répond-il à sa sonde ?"""
        probes = {"mock_source": "/health", "api": "/health", "alert_sink": "/health", "mlflow": "/health",
                  "prefect": "/api/health", "ui": "/_stcore/health"}  # fmt: skip
        try:
            return self.http.get(self.url(service, probes[service]), timeout=5).status_code == 200
        except httpx.HTTPError:
            return False

    def wait_healthy(self, service: str, timeout_s: float = 180) -> None:
        """Attend que le service réponde."""
        wait_for(lambda: self.healthy(service), timeout_s, f"{service} en ligne", interval_s=3)

    def wait_down(self, service: str, timeout_s: float = 60) -> None:
        """Attend que le service ne réponde plus."""
        wait_for(lambda: not self.healthy(service), timeout_s, f"{service} arrêté", interval_s=1)

    # ------------------------------------------------------------------ mock_source (API d'administration)

    def clock(self) -> date:
        """Date simulée de la source."""
        response = self.http.get(self.url("mock_source", "/admin/clock"))
        response.raise_for_status()
        return date.fromisoformat(response.json()["now"])

    def set_clock(self, day: date) -> None:
        """Règle l'horloge simulée (après un redémarrage de la source)."""
        response = self.http.post(self.url("mock_source", "/admin/clock"), json={"date": day.isoformat()})
        response.raise_for_status()

    def advance(self, days: int) -> ClockStep:
        """Fait passer `days` jours simulés."""
        clock = SimulationClock(self.url("mock_source"), self.config.ingestion.http_timeout_s)
        try:
            return clock.advance(days)
        finally:
            clock.close()

    def chaos(self, mode: str, failures: int | None = None, delay_s: float | None = None) -> dict[str, Any]:
        """Déclenche (ou arrête avec `off`) un incident de la source."""
        body: dict[str, Any] = {"mode": mode}
        if failures is not None:
            body["failures"] = failures
        if delay_s is not None:
            body["delay_s"] = delay_s
        response = self.http.post(self.url("mock_source", "/admin/chaos"), json=body)
        response.raise_for_status()
        return dict(response.json())

    # ------------------------------------------------------------------ API de prédiction

    def ready(self) -> tuple[int, dict[str, Any]]:
        """`GET /ready` : code HTTP et corps."""
        response = self.http.get(self.url("api", "/ready"))
        return response.status_code, dict(response.json())

    def model_info(self) -> dict[str, Any]:
        """`GET /model`."""
        response = self.http.get(self.url("api", "/model"))
        response.raise_for_status()
        return dict(response.json())

    def served_version(self) -> str | None:
        """Version que l'API sert en ce moment."""
        served = self.model_info().get("served")
        return str(served["version"]) if served else None

    def predict(self, body: Mapping[str, Any]) -> httpx.Response:
        """`POST /predict`, présenté comme client `scenarios`."""
        return self.http.post(self.url("api", "/predict"), json=dict(body), headers={"X-Client": CLIENT})

    # ------------------------------------------------------------------ service d'alertes

    def alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Historique des alertes (les plus récentes d'abord)."""
        response = self.http.get(self.url("alert_sink", "/alerts"), params={"limit": limit})
        response.raise_for_status()
        return list(response.json())

    def alert_health(self) -> dict[str, Any]:
        """`GET /health` du service d'alertes (Teams configuré ? silence ?)."""
        response = self.http.get(self.url("alert_sink", "/health"))
        response.raise_for_status()
        return dict(response.json())

    def silence(self, minutes: int, reason: str) -> dict[str, Any]:
        """Mode silence : les cartes sont gardées, rien ne part sur Teams."""
        response = self.http.post(self.url("alert_sink", "/silence"), json={"minutes": minutes, "reason": reason})
        response.raise_for_status()
        return dict(response.json())

    def unsilence(self) -> None:
        """Fin du mode silence."""
        self.http.delete(self.url("alert_sink", "/silence")).raise_for_status()

    def wait_alert(self, kind: str, timeout_s: float = 120, *, flow_run_id: str | None = None,
                   since: datetime | None = None) -> dict[str, Any]:  # fmt: skip
        """Attend une alerte de ce type (pour ce flow run, ou reçue depuis `since`)."""

        def probe() -> dict[str, Any] | None:
            for alert in self.alerts():
                if alert.get("kind") != kind:
                    continue
                if flow_run_id is not None and alert.get("flow_run_id") != flow_run_id:
                    continue
                at = datetime.fromisoformat(alert["at"]) if alert.get("at") else None
                if since is not None and (at is None or _aware(at) < since):
                    continue
                return alert
            return None

        target = f" du run {flow_run_id[:8]}" if flow_run_id else ""
        return wait_for(probe, timeout_s, f"alerte « {kind} »{target} reçue par alert_sink", interval_s=3)

    # ------------------------------------------------------------------ registre MLflow

    @property
    def registry(self) -> ModelRegistry:
        """Le registre du modèle (`atp-winner`)."""
        if self._registry is None:
            self._registry = ModelRegistry.connect(str(self.settings.mlflow_tracking_uri), self.train.registered_model)
        return self._registry

    def alias(self, alias: Alias) -> str | None:
        """Version désignée par un alias."""
        return self.registry.aliases().get(alias.value)

    def version_tags(self, version: str) -> dict[str, str]:
        """Tags d'une version."""
        return dict(self.registry.get(version).tags)

    def latest_version(self) -> str | None:
        """La dernière version enregistrée."""
        versions = self.registry.versions()
        return str(versions[0].version) if versions else None

    def version_count(self) -> int:
        """Nombre de versions enregistrées."""
        return len(self.registry.versions())

    # ------------------------------------------------------------------ base du projet (lecture)

    @property
    def engine(self) -> sa.Engine:
        """Connexion à la base du projet (compte `tennis`) ; tables créées si la base est neuve, comme les flows."""
        if self._engine is None:
            engine = sa.create_engine(self.settings.sqlalchemy_url(), pool_pre_ping=True)
            IngestionStore(engine).create_schema()
            PipelineEvents(engine).create_schema()
            self._engine = engine
        return self._engine

    def rows(self, sql: str, **params: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        """Lignes d'une requête de lecture."""
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(sa.text(sql), params).mappings()]

    def scalar(self, sql: str, **params: Any) -> Any:  # noqa: ANN401
        """Une valeur."""
        with self.engine.connect() as conn:
            return conn.execute(sa.text(sql), params).scalar()

    def watermark(self) -> date:
        """Date jusqu'à laquelle tout est ingéré."""
        return IngestionStore(self.engine).watermark(default=self.config.simulation.bootstrap_start)

    def events(self, kind: str, since: datetime) -> list[dict[str, Any]]:
        """Faits de `pipeline_events` d'un type, depuis `since` (payload décodé)."""
        rows = self.rows("SELECT * FROM pipeline_events WHERE kind = :k AND occurred_at >= :s ORDER BY occurred_at",
                         k=kind, s=since)  # fmt: skip
        for row in rows:
            row["payload"] = json.loads(row.get("payload") or "{}")
        return rows

    # ------------------------------------------------------------------ Prefect

    def flow_runs(self, flow_names: Iterable[str] = OUR_FLOWS, since: datetime | None = None) -> list[RunInfo]:
        """Les flow runs de nos flows (créés depuis `since`), du plus récent au plus ancien."""
        from prefect import get_client
        from prefect.client.schemas.filters import (
            FlowFilter,
            FlowFilterName,
            FlowRunFilter,
            FlowRunFilterExpectedStartTime,
        )
        from prefect.client.schemas.sorting import FlowRunSort

        names = list(flow_names)
        after = FlowRunFilterExpectedStartTime(after_=since) if since else None  # type: ignore[arg-type]
        run_filter = FlowRunFilter(expected_start_time=after) if after else None
        with get_client(sync_client=True) as client:
            flows = {f.id: f.name for f in client.read_flows(flow_filter=FlowFilter(name=FlowFilterName(any_=names)))}
            if not flows:
                return []
            runs = client.read_flow_runs(flow_filter=FlowFilter(name=FlowFilterName(any_=names)),
                                         flow_run_filter=run_filter, sort=FlowRunSort.EXPECTED_START_TIME_DESC,
                                         limit=50)  # fmt: skip
        return [_run_info(r, flows.get(r.flow_id, "?")) for r in runs]

    def flow_run(self, flow_run_id: str) -> RunInfo:
        """Un flow run par son identifiant."""
        from prefect import get_client

        with get_client(sync_client=True) as client:
            run = client.read_flow_run(UUID(flow_run_id))
            flow = client.read_flow(run.flow_id)
        return _run_info(run, flow.name)

    def task_runs(self, flow_run_id: str) -> list[TaskInfo]:
        """Les task runs d'un flow run (nom, essais, état)."""
        from prefect import get_client
        from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterId

        with get_client(sync_client=True) as client:
            tasks = client.read_task_runs(flow_run_filter=FlowRunFilter(id=FlowRunFilterId(any_=[UUID(flow_run_id)])))
        return [_task_info(task) for task in tasks]

    def pause_simulation(self) -> bool:
        """Met la simulation planifiée en pause (`make demo-stop`) ; vrai si elle tournait."""
        from prefect import get_client

        from tennis_mlops.orchestration.__main__ import SIMULATION

        with get_client(sync_client=True) as client:
            deployment = client.read_deployment_by_name(SIMULATION)
            if deployment.paused:
                return False
            client.pause_deployment(deployment.id)
        return True


def _run_info(run: Any, flow_name: str) -> RunInfo:  # noqa: ANN401
    state = run.state
    return RunInfo(str(run.id), run.name, flow_name, state.type.value if state else "?",
                   (state.name if state else None) or "?", run.created)  # fmt: skip


def _task_info(task: Any) -> TaskInfo:  # noqa: ANN401
    state, seconds = task.state, task.total_run_time.total_seconds() if task.total_run_time else None
    return TaskInfo(task.name, int(task.run_count), (state.name if state else None) or "?", seconds)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def now() -> datetime:
    """Maintenant, en UTC (comparable aux horodatages de la base et de Prefect)."""
    return datetime.now(UTC)


def names(items: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    """Les valeurs d'une clé, pour un message."""
    return [str(item.get(key)) for item in items]
