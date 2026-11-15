"""Mise en place de l'orchestration dans Prefect : work pool, déploiements, automations. Idempotent.

- work pool `tennis-pool` (process) : le worker de la stack y prend les runs et les exécute dans son conteneur ;
- `simulation-atp/planifiee` : toutes les `simulation_interval_s` secondes, +`simulation_days_per_run` jours puis
  ingestion ; en pause tant que `make demo` ne l'a pas lancé ;
- `entrainement-atp/sur-evenement` : une campagne à la fois, lancée par l'automation (ou à la main dans l'UI) ;
- `ingestion-atp/manuelle` : un passage d'ingestion depuis l'UI Prefect ;
- automation « tennis · réentraînement » : `tennis.training.requested` → `entrainement-atp/sur-evenement` ;
- automation « tennis · alertes » : état final `Failed`, `Crashed` ou `Dégradé` d'un de nos flows → `POST /prefect`
  du service d'alertes.

Relancé à chaque `make up` (service `orchestration-setup`) : il met à jour sans dupliquer et conserve la pause (ou
non) de la simulation, pour qu'un `make up` en pleine démo ne l'arrête pas.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, cast
from uuid import UUID

from prefect.blocks.webhook import Webhook
from prefect.client.orchestration import SyncPrefectClient
from prefect.client.schemas.actions import DeploymentScheduleCreate, WorkPoolCreate
from prefect.client.schemas.objects import ConcurrencyLimitStrategy, ConcurrencyOptions
from prefect.client.schemas.schedules import IntervalSchedule
from prefect.events.actions import CallWebhook, RunDeployment
from prefect.events.schemas.automations import AutomationCore, EventTrigger, Posture
from prefect.exceptions import ObjectNotFound
from prefect.types import SecretDict
from prefect.utilities.callables import parameter_schema
from prefect.workers.process import ProcessWorker
from pydantic import SecretStr

from tennis_mlops.config import ProjectConfig
from tennis_mlops.flows.ingest import DEGRADED_STATE, ingest_flow
from tennis_mlops.flows.simulation import simulation_flow
from tennis_mlops.flows.train import training_flow
from tennis_mlops.orchestration.trigger import DATASET_RESOURCE, TRAINING_REQUESTED

POOL = "tennis-pool"
SOURCE_DIR = "/app"  # le code dans l'image : le worker exécute les flows depuis là
TAGS = ["tennis-mlops"]
WEBHOOK_BLOCK = "tennis-alert-sink"
ALERTS_AUTOMATION = "tennis · alertes"
TRAINING_AUTOMATION = "tennis · réentraînement"
SUBFLOWS = ("scoring-atp", "feedback-atp")  # lancés par la simulation, sans déploiement à eux : surveillés aussi
ALERT_STATES = ("Failed", "Crashed", DEGRADED_STATE)
# Ce que l'automation transmet au service d'alertes : des identifiants, le service relit le reste (alerts/sink.py).
WEBHOOK_PAYLOAD = (
    '{"event_id": "{{ event.id }}", "event": "{{ event.event }}", "flow_run_id": "{{ event.resource.id }}"}'
)


@dataclass(frozen=True)
class DeploymentSpec:
    """Un déploiement à créer ou mettre à jour."""

    flow: Callable[..., Any]
    flow_name: str
    name: str
    entrypoint: str
    description: str
    interval_s: int | None = None
    paused_by_default: bool = False
    collision: ConcurrencyLimitStrategy = ConcurrencyLimitStrategy.ENQUEUE
    env: dict[str, str] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """`flow/déploiement`, comme dans l'UI et la CLI Prefect."""
        return f"{self.flow_name}/{self.name}"


def deployment_specs(config: ProjectConfig, training_workers: int) -> list[DeploymentSpec]:
    """Les trois déploiements du projet."""
    orchestration = config.orchestration
    return [
        DeploymentSpec(
            simulation_flow,
            "simulation-atp",
            "planifiee",
            "tennis_mlops.flows.simulation:simulation_flow",
            f"+{orchestration.simulation_days_per_run} jours simulés puis ingestion, toutes les "
            f"{orchestration.simulation_interval_s} s (make demo / make demo-stop)",
            interval_s=orchestration.simulation_interval_s,
            paused_by_default=True,
            collision=ConcurrencyLimitStrategy.CANCEL_NEW,  # un pas en retard est sauté, jamais empilé
        ),
        DeploymentSpec(
            training_flow,
            "entrainement-atp",
            "sur-evenement",
            "tennis_mlops.flows.train:training_flow",
            f"Campagne lancée par l'événement {TRAINING_REQUESTED} (≥ {orchestration.training_trigger_matches} "
            "nouveaux matchs validés) ; une à la fois",
            env={"PREFECT_TASKS_RUNNER_THREAD_POOL_MAX_WORKERS": str(training_workers)},
        ),
        DeploymentSpec(
            ingest_flow,
            "ingestion-atp",
            "manuelle",
            "tennis_mlops.flows.ingest:ingest_flow",
            "Un passage d'ingestion depuis l'UI (la simulation l'appelle elle-même en sous-flow)",
        ),
    ]


def ensure_pool(client: SyncPrefectClient) -> str:
    """Le work pool `process`, avec le gabarit de job par défaut du worker."""
    try:
        client.read_work_pool(POOL)
    except ObjectNotFound:
        template = ProcessWorker.get_default_base_job_template()
        client.create_work_pool(WorkPoolCreate(name=POOL, type="process", base_job_template=template,
                                               description="Worker de la stack tennis-mlops"))  # fmt: skip
        return f"work pool {POOL} créé"
    return f"work pool {POOL} : déjà là"


def ensure_deployment(client: SyncPrefectClient, spec: DeploymentSpec) -> tuple[UUID, str]:
    """Crée ou met à jour le déploiement ; garde l'état de pause d'un déploiement existant."""
    try:
        existing = client.read_deployment_by_name(spec.full_name)
    except ObjectNotFound:
        existing = None
    paused = existing.paused if existing is not None else spec.paused_by_default
    if existing is not None:  # le planning est recréé : pas de doublon après plusieurs `make up`
        for schedule in client.read_deployment_schedules(existing.id):
            client.delete_deployment_schedule(existing.id, schedule.id)
    schedules = (
        [DeploymentScheduleCreate(schedule=IntervalSchedule(interval=timedelta(seconds=spec.interval_s)), active=True)]
        if spec.interval_s
        else None
    )
    flow_id = client.create_flow_from_name(spec.flow_name)
    deployment_id = client.create_deployment(
        flow_id=flow_id,
        name=spec.name,
        description=spec.description,
        work_pool_name=POOL,
        entrypoint=spec.entrypoint,
        path=SOURCE_DIR,
        pull_steps=[{"prefect.deployments.steps.set_working_directory": {"directory": SOURCE_DIR}}],
        parameter_openapi_schema=parameter_schema(spec.flow).model_dump_for_openapi(),
        schedules=schedules,
        paused=paused,
        concurrency_limit=1,
        concurrency_options=ConcurrencyOptions(collision_strategy=spec.collision),
        job_variables={"env": spec.env} if spec.env else None,
        tags=TAGS,
        enforce_parameter_schema=True,
    )
    state = "en pause" if paused else "actif"
    every = f", toutes les {spec.interval_s} s" if spec.interval_s else ""
    return deployment_id, f"déploiement {spec.full_name} ({state}{every})"


def ensure_webhook(alert_sink_url: str) -> UUID:
    """Bloc `Webhook` vers `POST /prefect` du service d'alertes (adresse interne au réseau Docker)."""
    block = Webhook(
        method="POST",
        url=SecretStr(f"{alert_sink_url.rstrip('/')}/prefect"),
        headers=SecretDict({"Content-Type": "application/json"}),
    )
    return cast("UUID", block.save(WEBHOOK_BLOCK, overwrite=True))  # appel synchrone : `save` est « sync compatible »


def alerts_automation(webhook_id: UUID, flow_names: list[str]) -> AutomationCore:
    """État final en échec, arrêt brutal ou dégradé d'un de nos flows → service d'alertes."""
    return AutomationCore(
        name=ALERTS_AUTOMATION,
        description="Un incident (après les retries) = une carte : Failed, Crashed ou Dégradé → alert_sink",
        trigger=EventTrigger(
            match={"prefect.resource.id": "prefect.flow-run.*"},
            match_related={"prefect.resource.role": "flow", "prefect.resource.name": flow_names},
            expect={f"prefect.flow-run.{state}" for state in ALERT_STATES},
            posture=Posture.Reactive,
            threshold=1,
            within=timedelta(0),
        ),
        actions=[CallWebhook(block_document_id=webhook_id, payload=WEBHOOK_PAYLOAD)],
    )


def training_automation(deployment_id: UUID) -> AutomationCore:
    """Événement `tennis.training.requested` → une campagne d'entraînement."""
    return AutomationCore(
        name=TRAINING_AUTOMATION,
        description=f"{TRAINING_REQUESTED} (émis par l'ingestion) → entrainement-atp/sur-evenement",
        trigger=EventTrigger(
            match={"prefect.resource.id": DATASET_RESOURCE},
            expect={TRAINING_REQUESTED},
            posture=Posture.Reactive,
            threshold=1,
            within=timedelta(0),
        ),
        actions=[RunDeployment(source="selected", deployment_id=deployment_id)],
    )


def ensure_automation(client: SyncPrefectClient, automation: AutomationCore) -> str:
    """Remplace l'automation du même nom (une seule à la fois, toujours à jour)."""
    replaced = client.read_automations_by_name(automation.name)
    for old in replaced:
        client.delete_automation(old.id)
    client.create_automation(automation)
    return f"automation « {automation.name} » {'mise à jour' if replaced else 'créée'}"


def setup(client: SyncPrefectClient, config: ProjectConfig, training_workers: int, alert_sink_url: str) -> list[str]:
    """Tout mettre en place ; renvoie ce qui a été fait, ligne par ligne."""
    lines = [ensure_pool(client)]
    ids: dict[str, UUID] = {}
    specs = deployment_specs(config, training_workers)
    for spec in specs:
        ids[spec.flow_name], line = ensure_deployment(client, spec)
        lines.append(line)
    webhook = ensure_webhook(alert_sink_url)
    lines.append(f"bloc webhook {WEBHOOK_BLOCK} → {alert_sink_url.rstrip('/')}/prefect")
    lines.append(ensure_automation(client, training_automation(ids["entrainement-atp"])))
    watched = [s.flow_name for s in specs] + list(SUBFLOWS)
    lines.append(ensure_automation(client, alerts_automation(webhook, watched)))
    return lines
