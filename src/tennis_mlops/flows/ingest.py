"""Flow Prefect d'ingestion (mode pull) : de `mock_source` à PostgreSQL, à travers les contrôles qualité.

```
GET /admin/clock ─► amorçage 2010→2015 (une fois) ─► GET /tournaments (terminés depuis le watermark)
   └─► pour chaque tournoi : GET /tournaments/{id}/results ─► Great Expectations ─┬─► stockage (transaction)
                                                                                    └─► quarantaine
   └─► watermark avancé jusqu'au premier lot refusé (il sera retenté au prochain passage)
```

- **Une task par appel HTTP**, chacune avec ses retries (backoff) : seule une panne (`503`, timeout, connexion) est
  retentée ; un `4xx` ne l'est jamais.
- **Lot refusé** : archivé en quarantaine, aucun match stocké, flow en échec (`Failed`) → alerte. Les lots valides
  du même passage sont quand même stockés.
- **Source indisponible** après tous les essais : flow terminé à l'état « Dégradé ». Les données déjà ingérées restent
  utilisables, le watermark ne bouge pas, rien n'est perdu.

- **Journal et déclenchement** : chaque passage écrit ses faits marquants dans `pipeline_events` et, au-delà de
  `orchestration.training_trigger_matches` nouveaux matchs, émet l'événement `tennis.training.requested` auquel
  réagit l'automation qui lance l'entraînement (phase 6).

Lancement : `make ingest` (ou `python -m tennis_mlops.flows.ingest`).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from prefect import State, Task, flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.client.schemas.objects import TaskRun
from prefect.events import emit_event
from prefect.runtime import flow_run
from prefect.states import Completed
from pydantic import BaseModel

from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig, Settings
from tennis_mlops.data.errors import DataQualityError, SourceUnavailableError
from tennis_mlops.ingestion.archive import LotArchive
from tennis_mlops.ingestion.client import MockSourceClient, SourcePayload
from tennis_mlops.ingestion.lots import Lot, bootstrap_lot, tournament_lot
from tennis_mlops.ingestion.store import IngestionStore
from tennis_mlops.orchestration.events import EventKind, PipelineEvent, PipelineEvents, Severity
from tennis_mlops.orchestration.trigger import Emitter, TrainingTrigger, TriggerDecision
from tennis_mlops.validation.suites import RankGapReference
from tennis_mlops.validation.validator import LotValidator, ValidationReport

DEGRADED_STATE = "Dégradé"


class IngestionStatus(StrEnum):
    """Issue d'un passage d'ingestion."""

    OK = "ok"
    UP_TO_DATE = "up_to_date"
    QUARANTINE = "quarantine"
    DEGRADED = "degraded"


class IngestionReport(BaseModel):
    """Bilan d'un passage, renvoyé par le flow et affiché par la ligne de commande."""

    status: IngestionStatus
    simulated_now: date | None
    watermark_before: date
    watermark_after: date
    lots_valid: list[str] = []
    lots_quarantined: list[str] = []
    lots_skipped: int = 0
    rows_inserted: int = 0
    warnings: list[str] = []
    detail: str | None = None
    training_requested: bool = False

    def summary(self) -> str:
        """Une ligne pour les logs."""
        return (
            f"{self.status.value} : {len(self.lots_valid)} lot(s) stocké(s), {len(self.lots_quarantined)} en "
            f"quarantaine, {self.rows_inserted} match(s) ajouté(s), watermark {self.watermark_before} → "
            f"{self.watermark_after}" + (" ; réentraînement demandé" if self.training_requested else "")
        )


# --------------------------------------------------------------------------- dépendances (injectables)


@dataclass
class IngestionDependencies:
    """Ce dont le flow a besoin ; les tests en fournissent une version en mémoire."""

    client: MockSourceClient
    store: IngestionStore
    validator: LotValidator
    archive: LotArchive

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.client.close()


def default_dependencies(config: ProjectConfig, settings: Settings, root: Path) -> IngestionDependencies:
    """Dépendances réelles : source HTTP, PostgreSQL, Great Expectations sur disque."""
    state_dir = config.ingestion.resolved_state_dir(root)
    return IngestionDependencies(
        client=MockSourceClient(str(settings.mock_source_url), config.ingestion.http_timeout_s),
        store=IngestionStore.from_url(settings.sqlalchemy_url()),
        validator=LotValidator(state_dir / "gx"),
        archive=LotArchive(state_dir),
    )


DependencyFactory = Callable[[ProjectConfig, Settings, Path], IngestionDependencies]
dependency_factory: DependencyFactory = default_dependencies
training_event_emitter: Emitter = emit_event  # remplacé dans les tests


# --------------------------------------------------------------------------- tasks : un appel HTTP = une task


def retry_on_outage(task: Task[..., Any], task_run: TaskRun, state: State[Any]) -> bool:
    """Ne retente que les pannes de la source ; une requête refusée (4xx) échoue tout de suite."""
    try:
        state.result()
    except SourceUnavailableError:
        return True
    except Exception:
        return False
    return False


@task(name="GET /admin/clock", cache_policy=NO_CACHE)
def fetch_clock(client: MockSourceClient) -> date:
    """Date simulée de la source."""
    return client.clock().simulated_now


@task(name="GET /results (amorçage)", cache_policy=NO_CACHE)
def fetch_bootstrap(client: MockSourceClient, since: date, until: date) -> SourcePayload:
    """Historique initial, en un seul appel."""
    return client.results(since, until)


@task(name="GET /tournaments", task_run_name="GET /tournaments since={since}", cache_policy=NO_CACHE)
def fetch_completed_tournaments(client: MockSourceClient, since: date, until: date) -> list[dict[str, Any]]:
    """Tournois terminés depuis le watermark, du plus ancien au plus récent."""
    tournaments: list[dict[str, Any]] = client.tournaments(since, until).data.get("tournaments", [])
    return sorted(tournaments, key=lambda t: (t["end_date"], t["tourney_id"]))


@task(
    name="GET /tournaments/{id}/results",
    task_run_name="GET /tournaments/{tourney_id}/results",
    cache_policy=NO_CACHE,
)
def fetch_tournament(client: MockSourceClient, tourney_id: str) -> SourcePayload:
    """Un lot : les résultats d'un tournoi terminé."""
    return client.tournament_results(tourney_id)


# --------------------------------------------------------------------------- tasks : contrôle et stockage


@task(name="Great Expectations", task_run_name="contrôles {lot.lot_id}", cache_policy=NO_CACHE)
def validate_lot(
    validator: LotValidator, lot: Lot, reference: RankGapReference | None, pause_s: float
) -> ValidationReport:
    """Suite bloquante + surveillance de dérive."""
    time.sleep(pause_s)
    report = validator.validate(lot, reference)
    logger = get_run_logger()
    for check in report.failures:
        logger.error("✗ %s — %s", lot.lot_id, check.describe())
    for check in report.warnings:
        logger.warning("dérive %s — %s", lot.lot_id, check.describe())
    return report


@task(name="stockage", task_run_name="stockage {lot.lot_id}", cache_policy=NO_CACHE)
def store_lot(deps: IngestionDependencies, lot: Lot, report: ValidationReport, pause_s: float) -> int:
    """Archive brute + écriture en base (une transaction)."""
    time.sleep(pause_s)
    folder = deps.archive.save(lot, report)
    inserted = deps.store.save_valid_lot(lot, report, str(folder), _flow_run_id())
    get_run_logger().info("%s : %d/%d match(s) ajouté(s)", lot.lot_id, inserted, lot.rows)
    return inserted


@task(name="quarantaine", task_run_name="quarantaine {lot.lot_id}", cache_policy=NO_CACHE)
def quarantine_lot(deps: IngestionDependencies, lot: Lot, report: ValidationReport) -> str:
    """Lot refusé : archivé à part et tracé, rien n'est stocké."""
    folder = deps.archive.save(lot, report)
    run_id = _flow_run_id()
    deps.store.save_quarantined_lot(lot, report, str(folder), run_id)
    get_run_logger().error("%s en quarantaine : %s", lot.lot_id, folder)
    PipelineEvents(deps.store.engine).record(quarantine_event(lot, report, folder, run_id))
    return str(folder)


def quarantine_event(lot: Lot, report: ValidationReport, folder: Path, flow_run_id: str | None) -> PipelineEvent:
    """Ce que la carte « lot refusé » affiche : tournoi, volumes, contrôles en échec (structurés)."""
    names = lot.frame["tourney_name"].dropna().astype(str).tolist() if "tourney_name" in lot.frame else []
    payload = {
        "lot_id": lot.lot_id,
        "tourney_name": names[0] if names else None,
        "tourney_level": lot.tourney_level,
        "end_date": lot.end_date,
        "rows_received": lot.rows,
        "rows_expected": lot.expected_rows,
        "archive": "/".join(folder.parts[-3:]) + "/",  # state/quarantine/<lot>/ : lisible sur tous les postes
        "failed_checks": [asdict(check) for check in report.failures],
    }
    return PipelineEvent(
        EventKind.LOT_QUARANTINED,
        Severity.FAILURE,
        f"Lot {lot.lot_id} en quarantaine",
        " ; ".join(check.describe() for check in report.failures),
        lot.simulated_now,
        flow_run_id,
        lot.lot_id,
        payload=payload,
    )


@task(name="journal et déclenchement de l'entraînement", cache_policy=NO_CACHE)
def follow_up(store: IngestionStore, config: ProjectConfig, report: IngestionReport) -> TriggerDecision:
    """Faits marquants dans `pipeline_events`, puis demande d'entraînement si assez de nouveaux matchs."""
    events, run_id = PipelineEvents(store.engine), _flow_run_id()  # lots refusés : journalisés par `quarantine_lot`
    if report.lots_valid:
        events.record(
            PipelineEvent(
                EventKind.INGESTION_COMPLETED,
                Severity.INFO,
                f"Ingestion : {len(report.lots_valid)} lot(s), {report.rows_inserted} match(s)",
                report.summary(),
                report.simulated_now,
                run_id,
                payload={"lots": report.lots_valid, "rows_inserted": report.rows_inserted},
            )
        )
    trigger = TrainingTrigger(store, events, config.orchestration.training_trigger_matches, training_event_emitter)
    decision = trigger.check(run_id, report.simulated_now)
    get_run_logger().info(decision.summary())
    return decision


def _record_degraded(store: IngestionStore, report: IngestionReport) -> None:
    PipelineEvents(store.engine).record(
        PipelineEvent(
            EventKind.INGESTION_DEGRADED,
            Severity.WARNING,
            "Ingestion dégradée : source injoignable",
            report.detail or "",
            flow_run_id=_flow_run_id(),
        )
    )


def _flow_run_id() -> str | None:
    run_id = flow_run.get_id()
    return str(run_id) if run_id else None


# --------------------------------------------------------------------------- flow


@flow(name="ingestion-atp")
def ingest_flow(config_path: str | None = None) -> IngestionReport | State[Any]:
    """Ingère les tournois terminés depuis le dernier passage, en contrôlant chaque lot."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = ProjectConfig.from_yaml(path)
    with dependency_factory(config, Settings(), path.resolve().parents[1]) as deps:
        deps.store.create_schema()
        watermark = deps.store.watermark(default=config.simulation.bootstrap_start)
        try:
            report = _ingest(config, deps, watermark)
        except SourceUnavailableError as exc:
            report = IngestionReport(
                status=IngestionStatus.DEGRADED,
                simulated_now=None,
                watermark_before=watermark,
                watermark_after=watermark,
                detail=str(exc),
            )
            get_run_logger().warning("Source indisponible après %d essais : %s", config.ingestion.retries + 1, exc)
            _record_degraded(deps.store, report)
            return Completed(name=DEGRADED_STATE, message=f"Source indisponible : {exc}", data=report)
        decision = follow_up(deps.store, config, report)
        report.training_requested = decision.requested
    get_run_logger().info(report.summary())
    if report.status is IngestionStatus.QUARANTINE:
        msg = f"lot(s) en quarantaine : {', '.join(report.lots_quarantined)} — {report.summary()}"
        raise DataQualityError(msg)
    return report


def _ingest(config: ProjectConfig, deps: IngestionDependencies, watermark: date) -> IngestionReport:
    options: dict[str, Any] = {
        "retries": config.ingestion.retries,
        "retry_delay_seconds": list(config.ingestion.retry_delays_s),
        "retry_condition_fn": retry_on_outage,
    }
    pause = config.demo.sleep_s
    simulation = config.simulation
    now = fetch_clock.with_options(**options)(deps.client)
    report = IngestionReport(
        status=IngestionStatus.OK, simulated_now=now, watermark_before=watermark, watermark_after=watermark
    )
    since = watermark

    if since < simulation.replay_start and since < now:
        until = min(now, simulation.replay_start)
        lot = bootstrap_lot(fetch_bootstrap.with_options(**options)(deps.client, since, until), since, until)
        if not _process(deps, lot, None, pause, report):
            report.status = IngestionStatus.QUARANTINE
            return report  # sans historique valide, inutile d'aller plus loin
        since = until

    tournaments = fetch_completed_tournaments.with_options(**options)(deps.client, since, now) if since < now else []
    already_valid = deps.store.valid_lot_ids()
    first_refused: date | None = None
    for tournament in tournaments:
        if tournament["tourney_id"] in already_valid:
            report.lots_skipped += 1
            continue
        lot = tournament_lot(fetch_tournament.with_options(**options)(deps.client, tournament["tourney_id"]))
        reference = deps.store.rank_gap_reference(
            str(tournament["tourney_level"]),
            date.fromisoformat(tournament["start_date"]),
            config.ingestion.drift_min_history,
            config.ingestion.drift_quantiles,
        )
        if not _process(deps, lot, reference, pause, report) and first_refused is None:
            first_refused = date.fromisoformat(tournament["end_date"])

    report.watermark_after = first_refused or max(since, now)
    deps.store.set_watermark(report.watermark_after)
    if report.lots_quarantined:
        report.status = IngestionStatus.QUARANTINE
    elif not report.lots_valid:
        report.status = IngestionStatus.UP_TO_DATE
    return report


def _process(
    deps: IngestionDependencies, lot: Lot, reference: RankGapReference | None, pause: float, report: IngestionReport
) -> bool:
    verdict = validate_lot(deps.validator, lot, reference, pause)
    report.warnings += [f"{lot.lot_id} : {w.describe()}" for w in verdict.warnings]
    if verdict.success:
        report.rows_inserted += store_lot(deps, lot, verdict, pause)
        report.lots_valid.append(lot.lot_id)
        return True
    quarantine_lot(deps, lot, verdict)
    report.lots_quarantined.append(lot.lot_id)
    return False


# --------------------------------------------------------------------------- ligne de commande

EXIT_OK, EXIT_QUARANTINE, EXIT_DEGRADED = 0, 1, 2


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m tennis_mlops.flows.ingest` : un passage d'ingestion ; code de sortie selon l'issue."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.flows.ingest", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    state = ingest_flow(str(args.config), return_state=True)
    if state.is_failed():
        return EXIT_QUARANTINE
    result = state.result(raise_on_failure=False)
    report = result if isinstance(result, IngestionReport) else None
    if report is not None:
        print(report.summary())
    return EXIT_DEGRADED if state.name == DEGRADED_STATE else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
