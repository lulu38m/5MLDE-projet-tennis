"""Ligne de commande du registre : `python -m tennis_mlops.registry status|promote|rollback|trace`.

- `status`   : versions, alias, cycle de vie et décision de chacune (`make registry`)
- `promote`  : met une version en production à la main, raison obligatoire (`make promote VERSION=3 REASON="…"`)
- `rollback` : `production` ← `previous` (`make rollback REASON="…"`)
- `trace`    : la chaîne version → run MLflow → campagne → run Prefect → lots → source (`make trace VERSION=3`)

Ne dépend que du serveur MLflow. Une promotion ou un rollback est ensuite annoncé (best effort) : une ligne dans
`pipeline_events` et le service d'alertes prévenu (carte Teams). Code de sortie 1 si l'opération est refusée ou si
MLflow ne répond pas ; jamais à cause de l'annonce.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mlflow.entities.model_registry import ModelVersion

from tennis_mlops.alerts.client import AlertSinkClient
from tennis_mlops.config import PROJECT_ROOT, Settings
from tennis_mlops.data.lockfile import DataLock
from tennis_mlops.orchestration.decisions import DecisionAnnouncer
from tennis_mlops.orchestration.events import PipelineEvents
from tennis_mlops.registry.events import EventKind, RegistryEvent
from tennis_mlops.registry.journal import DecisionJournal, journal_experiment
from tennis_mlops.registry.model_registry import ModelRegistry, UnknownVersionError
from tennis_mlops.registry.promotion import Promoter, PromotionError
from tennis_mlops.registry.rollback import Rollback, RollbackError
from tennis_mlops.registry.tagging import Alias
from tennis_mlops.training.config import DEFAULT_TRAIN_CONFIG_PATH, TrainConfig
from tennis_mlops.training.tracking import MlflowTracker, TrackingError

EXIT_OK, EXIT_REFUSED = 0, 1
DATA_LOCK_PATH = PROJECT_ROOT / "configs" / "data.lock.json"


def default_announcer() -> DecisionAnnouncer:
    """Base du projet et service d'alertes de la stack (adresses de `.env`)."""
    settings = Settings()
    return DecisionAnnouncer(PipelineEvents.from_url(settings.sqlalchemy_url()),
                             AlertSinkClient(str(settings.alert_sink_url)))  # fmt: skip


announcer_factory: Callable[[], DecisionAnnouncer] = default_announcer  # remplacé dans les tests


@dataclass(frozen=True)
class Connection:
    """Le registre du modèle, son journal de décisions et l'adresse du serveur."""

    registry: ModelRegistry
    journal: DecisionJournal
    tracking_uri: str

    @classmethod
    def open(cls, tracking_uri: str, model: str) -> Connection:
        """Vérifie que MLflow répond, puis ouvre le registre et le journal."""
        tracker = MlflowTracker(tracking_uri, journal_experiment(model))
        return cls(ModelRegistry(tracker.client, model), DecisionJournal(tracker), tracker.tracking_uri)

    def run_url(self, run_id: str) -> str:
        """Lien vers un run dans l'UI MLflow."""
        experiment = self.registry.client.get_run(run_id).info.experiment_id
        return f"{self.tracking_uri}/#/experiments/{experiment}/runs/{run_id}"


# --------------------------------------------------------------------------- status


def status_lines(registry: ModelRegistry) -> list[str]:
    """Tableau texte des versions, de la plus récente à la plus ancienne."""
    versions = registry.versions()
    if not versions:
        return [f"{registry.name} : aucune version enregistrée (lancer `make train`)"]
    aliases = registry.aliases()
    by_version: dict[str, list[str]] = {}
    for alias, version in aliases.items():
        by_version.setdefault(version, []).append(alias)
    header = " · ".join(f"{a} : v{aliases[a.value]}" for a in Alias if a.value in aliases) or "aucun alias posé"
    rows = [("version", "alias", "lifecycle", "règle", "log loss test", "données jusqu'au", "enregistrée")]
    for v in versions:
        tags = v.tags
        rows.append(
            (
                f"v{v.version}",
                ", ".join(sorted(by_version.get(str(v.version), []))) or "—",
                tags.get("lifecycle", "?"),
                tags.get("gate_status", "—"),
                tags.get("test.log_loss", "—"),
                tags.get("data_window_end", tags.get("data_end", "—")),
                _date(v.creation_timestamp),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    table = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip() for row in rows]
    production = registry.by_alias(Alias.PRODUCTION)
    lines = [f"{registry.name} — {header}", "", *table]
    if production is not None:
        lines += ["", f"production : v{production.version} — {production.tags.get('gate_reason', '')}".rstrip(" —")]
    return lines


def _date(timestamp_ms: int | None) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M") if timestamp_ms else "?"


# --------------------------------------------------------------------------- trace


def trace_lines(connection: Connection, version: ModelVersion, lock_path: Path = DATA_LOCK_PATH) -> list[str]:
    """La chaîne de traçabilité d'une version, maillon par maillon."""
    registry, tags = connection.registry, version.tags
    aliases = sorted(a for a, v in registry.aliases().items() if v == str(version.version))
    campaign = tags.get("campaign_run_id", "")
    decided = (
        f"{tags.get('gate_status', 'pas de décision')} ({tags.get('decided_by', '?')}, {tags.get('decided_at', '?')})"
    )
    lines = [
        f"{registry.name} v{version.version} — alias : {', '.join(aliases) or 'aucun'} · lifecycle : "
        f"{tags.get('lifecycle', '?')}",
        "",
        "0. prédiction  (phase 7) chaque ligne `predictions` portera model_version + request_id → cette version",
        f"1. version     décision {decided} — {tags.get('gate_reason', '')}".rstrip(" —"),
    ]
    for key in ("rollback_reason", "promotion_reason"):
        if tags.get(key):
            lines.append(f"                {key} : {tags[key]}")
    lines += [
        f"2. run MLflow  {connection.run_url(str(version.run_id))}",
        f"                candidat {tags.get('candidate', '?')} · test log loss {tags.get('test.log_loss', '?')} · "
        f"config {tags.get('config_hash', '?')} · features v{tags.get('features_version', '?')}",
        f"                git {_short(tags.get('git_sha', '?'))} (modifications non commitées : "
        f"{tags.get('git_dirty', '?')}) · uv.lock {_short(tags.get('uv_lock_sha256', '?'))}",
        f"3. campagne    {connection.run_url(campaign) if campaign else '—'}",
        f"4. run Prefect {tags.get('prefect_flow_run_url') or '—'}",
        f"5. données     {tags.get('lots_valid', '?')} lots acceptés par la suite GX "
        f"v{tags.get('gx_suite_version', '?')} ({tags.get('lots_quarantined', '?')} en quarantaine exclus) · "
        f"dernier lot {tags.get('last_valid_lot', '?')}",
        f"                fenêtre {tags.get('data_window_start', '?')} → {tags.get('data_window_end', '?')} · test "
        f"{tags.get('test_window', '?')} · horloge simulée {tags.get('simulated_now', '?')}",
        f"6. source      {_source(tags.get('dataset_version', ''), lock_path)}",
    ]
    return lines


def _short(value: str) -> str:
    """Une empreinte hexadécimale abrégée ; tout autre texte tel quel."""
    return value[:12] if re.fullmatch(r"[0-9a-f]{40,64}", value) else value


def _source(dataset_version: str, lock_path: Path) -> str:
    if not lock_path.exists():
        return f"dataset_version {dataset_version or '?'} (verrou {lock_path.name} absent)"
    lock = DataLock.load(lock_path)
    revisions = ", ".join(f"{name} {rev[:12]}" for name, rev in sorted(lock.revisions.items()))
    match = "conforme au verrou" if dataset_version == lock.dataset_version else f"verrou actuel {lock.dataset_version}"
    return (
        f"dataset_version {dataset_version or '?'} ({match}) · révisions : {revisions} · SHA-256 dans {lock_path.name}"
    )


# --------------------------------------------------------------------------- commandes


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée ; code de sortie 0 si l'opération a eu lieu, 1 sinon."""
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")  # nos propres messages suffisent
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.registry", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--tracking-uri", default=None, help="serveur MLflow (défaut : TENNIS_MLFLOW_TRACKING_URI)")
    parser.add_argument("--model", default=None, help="modèle enregistré (défaut : configs/train.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="versions, alias et décisions")
    promote = commands.add_parser("promote", help="mise en production manuelle")
    promote.add_argument("--version", required=True)
    promote.add_argument("--reason", required=True)
    rollback = commands.add_parser("rollback", help="production ← previous")
    rollback.add_argument("--reason", required=True)
    trace = commands.add_parser("trace", help="chaîne de traçabilité d'une version")
    trace.add_argument("--version", default=None, help="défaut : la version en production")
    args = parser.parse_args(argv)

    uri = args.tracking_uri or str(Settings().mlflow_tracking_uri)
    model = args.model or TrainConfig.from_yaml(DEFAULT_TRAIN_CONFIG_PATH).registered_model
    try:
        connection = Connection.open(uri, model)
        lines = _run(connection, args)
    except (TrackingError, PromotionError, RollbackError, UnknownVersionError) as exc:
        print(f"refusé : {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print("\n".join(lines))
    return EXIT_OK


def _run(connection: Connection, args: argparse.Namespace) -> list[str]:
    registry = connection.registry
    if args.command == "status":
        return status_lines(registry)
    if args.command == "promote":
        event = Promoter(registry).promote_manually(str(args.version), args.reason)
        return [event.summary(), *_journal(connection, event), *_announce(registry, event)]
    if args.command == "rollback":
        event = Rollback(registry).run(args.reason)
        lines = [event.summary(), "L'API suit l'alias `production` : elle sert cette version sans redéploiement."]
        return lines + _journal(connection, event) + _announce(registry, event)
    version = registry.get(str(args.version)) if args.version else registry.by_alias(Alias.PRODUCTION)
    if version is None:
        msg = f"aucune version de {registry.name} en production : préciser VERSION=…"
        raise UnknownVersionError(msg)
    return trace_lines(connection, version)


def _announce(registry: ModelRegistry, event: RegistryEvent) -> list[str]:
    metrics: dict[str, str | None] = {}
    if event.kind is EventKind.ROLLBACK:  # la carte dit si un 2e rollback reste possible
        following = registry.by_alias(Alias.PREVIOUS)
        metrics["previous_after"] = str(following.version) if following else None
    return announcer_factory().announce(event, metrics)


def _journal(connection: Connection, event: RegistryEvent) -> list[str]:
    problem = connection.journal.try_record(event)
    return [f"attention : {problem}"] if problem else []


if __name__ == "__main__":
    raise SystemExit(main())
