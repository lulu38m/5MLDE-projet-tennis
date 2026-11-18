"""`python -m tennis_mlops.orchestration setup|start|stop|status` : la chaîne automatique dans Prefect.

- `setup`  : work pool, déploiements, automations (idempotent ; lancé à chaque `make up` par `orchestration-setup`)
- `start`  : `make demo` — la simulation planifiée reprend, un premier pas part tout de suite
- `stop`   : `make demo-stop` — la simulation est mise en pause (un run en cours va jusqu'au bout)
- `status` : `make demo-status` — déploiements, worker, pas planifiés de la simulation, derniers runs

Utilise `PREFECT_API_URL` (exporté par le Makefile, ou fixé dans les conteneurs).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from prefect import get_client
from prefect.client.orchestration import SyncPrefectClient
from prefect.client.schemas.filters import (
    DeploymentFilter,
    DeploymentFilterId,
    DeploymentFilterTags,
    FlowRunFilter,
    FlowRunFilterExpectedStartTime,
    FlowRunFilterStartTime,
)
from prefect.client.schemas.sorting import FlowRunSort
from prefect.exceptions import ObjectNotFound, PrefectHTTPStatusError

from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig, Settings
from tennis_mlops.orchestration.setup import POOL, TAGS, setup
from tennis_mlops.training.config import DEFAULT_TRAIN_CONFIG_PATH, TrainConfig

SIMULATION = "simulation-atp/planifiee"
EXIT_OK, EXIT_FAILED = 0, 1
WORKER_FRESH_S = 60
RECENT_RUNS = 10
PLANNED_RUNS = 8


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée ; code de sortie 1 si Prefect ne répond pas ou si l'orchestration n'est pas en place."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.orchestration", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("setup", help="work pool, déploiements, automations")
    init.add_argument("--alert-sink-url", default=None, help="adresse du service d'alertes vue par Prefect")
    commands.add_parser("start", help="lance la simulation planifiée (make demo)")
    commands.add_parser("stop", help="met la simulation en pause (make demo-stop)")
    commands.add_parser("status", help="état de la chaîne (make demo-status)")
    args = parser.parse_args(argv)
    try:
        with get_client(sync_client=True) as client:
            lines = _run(client, args)
    except (ObjectNotFound, PrefectHTTPStatusError) as exc:
        print(f"refusé : {exc} — l'orchestration est-elle en place ? (make up)", file=sys.stderr)
        return EXIT_FAILED
    except OSError as exc:
        print(f"Prefect injoignable ({type(exc).__name__}) : la stack tourne-t-elle ? (make up)", file=sys.stderr)
        return EXIT_FAILED
    print("\n".join(lines))
    return EXIT_OK


def _run(client: SyncPrefectClient, args: argparse.Namespace) -> list[str]:
    settings = Settings()
    if args.command == "setup":
        config = ProjectConfig.from_yaml(DEFAULT_CONFIG_PATH)
        workers = TrainConfig.from_yaml(DEFAULT_TRAIN_CONFIG_PATH).workers
        return setup(client, config, workers, args.alert_sink_url or str(settings.alert_sink_url))
    simulation = client.read_deployment_by_name(SIMULATION)
    if args.command == "start":
        client.resume_deployment(simulation.id)
        run = client.create_flow_run_from_deployment(simulation.id, tags=["demo"])
        ui = str(settings.prefect_ui_url).rstrip("/")
        return [
            f"simulation lancée : premier pas maintenant ({run.name}), puis selon le planning du déploiement",
            *_worker_lines(client),
            f"suivi : {ui}/deployments/deployment/{simulation.id}",
            "arrêt : make demo-stop",
        ]
    if args.command == "stop":
        client.pause_deployment(simulation.id)
        return ["simulation en pause : plus de nouveau pas (un run en cours se termine) ; reprise : make demo"]
    return _status(client)


def _worker_lines(client: SyncPrefectClient) -> list[str]:
    workers = client.read_workers_for_work_pool(POOL)
    now = datetime.now(UTC)
    alive = [
        w for w in workers if w.last_heartbeat_time and (now - w.last_heartbeat_time).total_seconds() < WORKER_FRESH_S
    ]
    if alive:
        return [f"worker : {', '.join(w.name for w in alive)} ({POOL})"]
    return [f"attention : aucun worker actif sur {POOL} — les runs attendront (make up, puis make logs)"]


def _planned_lines(client: SyncPrefectClient) -> list[str]:
    """Les derniers pas prévus de la simulation, démarrés ou non.

    Un pas en retard, annulé ou qui attend une place de concurrence n'a pas d'heure de début : il n'apparaît donc pas
    dans « derniers runs », et c'est justement lui qu'on cherche quand l'horloge ne bouge plus.
    """
    try:
        simulation = client.read_deployment_by_name(SIMULATION)
    except ObjectNotFound:
        return []
    limit = simulation.global_concurrency_limit
    slots = f", concurrence {limit.active_slots}/{limit.limit} place(s) occupée(s)" if limit else ""
    lines = [f"pas de la simulation ({'en pause' if simulation.paused else 'actif'}{slots}) :"]
    runs = client.read_flow_runs(
        deployment_filter=DeploymentFilter(id=DeploymentFilterId(any_=[simulation.id])),
        flow_run_filter=FlowRunFilter(
            expected_start_time=FlowRunFilterExpectedStartTime(before_=datetime.now(UTC)),  # type: ignore[arg-type]
        ),
        sort=FlowRunSort.EXPECTED_START_TIME_DESC,
        limit=PLANNED_RUNS,
    )
    for run in runs:
        planned = run.expected_start_time.astimezone().strftime("%d/%m %H:%M:%S") if run.expected_start_time else "?"
        origin = "planifié" if run.auto_scheduled else "make demo"
        lines.append(f"  {planned}  {run.state_name or '?':<24} {origin:<9} {run.name}")
    if not runs:
        lines.append("  aucun pas prévu jusqu'ici")
    return lines


def _status(client: SyncPrefectClient) -> list[str]:
    flows: dict[UUID, str] = {}

    def flow_name(flow_id: UUID) -> str:
        if flow_id not in flows:
            flows[flow_id] = client.read_flow(flow_id).name
        return flows[flow_id]

    deployments = client.read_deployments(deployment_filter=DeploymentFilter(tags=DeploymentFilterTags(all_=TAGS)))
    lines = ["déploiements :"]
    for deployment in sorted(deployments, key=lambda d: flow_name(d.flow_id)):
        name = f"{flow_name(deployment.flow_id)}/{deployment.name}"
        lines.append(f"  {name:<32} {'en pause' if deployment.paused else 'actif'}")
    lines += _worker_lines(client)
    lines += _planned_lines(client)
    started = FlowRunFilter(start_time=FlowRunFilterStartTime(is_null_=False))  # PostgreSQL trie les NULL en tête
    runs = client.read_flow_runs(flow_run_filter=started, sort=FlowRunSort.START_TIME_DESC, limit=RECENT_RUNS)
    lines.append("derniers runs :")
    for run in runs:
        when = run.start_time.astimezone().strftime("%d/%m %H:%M:%S") if run.start_time else "?"
        lines.append(f"  {when}  {run.state_name or '?':<10} {flow_name(run.flow_id):<18} {run.name}")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
