"""Flow Prefect de simulation : un pas de temps simulé, puis ingestion, feedback et scoring. Planifié par `make demo`.

```
POST /admin/clock/advance (+7 jours) ─► ingestion-atp ─┬─► lots validés ─► assez de nouveaux matchs ?
                                                         │                     └─► événement ─► entraînement
                                                         └─► quarantaine / dégradé ─► alerte
                                     ─► feedback-atp : vrais résultats des matchs prédits ─► POST /feedback
                                     ─► scoring-atp  : matchs de la semaine à venir      ─► POST /predict/batch
```

- **Un incident = une alerte** : l'issue de l'ingestion (quarantaine, dégradé) est portée par le sous-flow
  `ingestion-atp`, qui déclenche sa propre alerte ; la simulation, elle, se termine normalement et le dit dans son
  bilan. Sinon chaque incident d'ingestion produirait deux cartes.
- **Horloge injoignable** : le pas est sauté mais l'ingestion tourne quand même ; c'est elle qui constate la panne
  (état « Dégradé », une alerte).
- **Fin du rejeu** (`simulation.replay_end`) : plus rien à avancer ; le flow le dit et ne fait rien d'autre.
- **Feedback puis scoring** (phase 7), en sous-flows eux aussi : d'abord les résultats que l'ingestion vient de
  valider, puis les prédictions de la semaine à venir. Si la source est injoignable (ingestion « Dégradé »), le
  scoring, qui en dépend, n'est pas lancé : la panne a déjà son alerte.

Lancement : planifié par le déploiement `simulation-atp` (`make demo`), ou `python -m tennis_mlops.flows.simulation`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from prefect import flow, get_client, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.exceptions import PrefectException
from prefect.runtime import flow_run
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig, Settings
from tennis_mlops.data.errors import SourceUnavailableError
from tennis_mlops.flows.feedback import feedback_flow
from tennis_mlops.flows.ingest import DEGRADED_STATE, IngestionReport, ingest_flow
from tennis_mlops.flows.scoring import scoring_flow
from tennis_mlops.ingestion.store import IngestionStore
from tennis_mlops.orchestration.clock import ClockStep, ReplayFinishedError, SimulationClock

FLOW_NAME = "simulation-atp"


class SimulationReport(BaseModel):
    """Bilan d'un pas de simulation."""

    days: int
    caught_up: date | None = None  # l'horloge de la source était en retard sur la base (redémarrée) et repartait d'ici
    previous: date | None = None
    now: date | None = None
    new_matches: int = 0
    tournaments: list[str] = []
    ingestion: str = "non lancée"  # nom de l'état final du sous-flow : Completed, Failed, Dégradé…
    ingestion_summary: str = ""
    feedback: str = "non lancé"
    scoring: str = "non lancé"
    finished: bool = False
    detail: str = ""

    def summary(self) -> str:
        """Une ligne pour les logs."""
        if self.finished:
            return f"fin du rejeu : {self.detail}"
        clock = f"{self.previous} → {self.now}, {self.new_matches} match(s) terminé(s)" if self.now else self.detail
        if self.caught_up:
            clock = f"horloge recalée depuis {self.caught_up}, {clock}"
        ingestion = f"ingestion {self.ingestion}" + (f" ({self.ingestion_summary})" if self.ingestion_summary else "")
        return f"pas de {self.days} jour(s) : {clock} ; {ingestion} ; feedback : {self.feedback} ; " + (
            f"scoring : {self.scoring}"
        )


ClockFactory = Callable[[Settings, ProjectConfig], SimulationClock]


def default_clock(settings: Settings, config: ProjectConfig) -> SimulationClock:
    """L'horloge de `mock_source` (adresse de `TENNIS_MOCK_SOURCE_URL`)."""
    return SimulationClock(str(settings.mock_source_url), config.ingestion.http_timeout_s)


clock_factory: ClockFactory = default_clock  # remplacé dans les tests


def default_store(settings: Settings) -> IngestionStore:
    """La base du projet (pour le watermark de l'ingestion)."""
    return IngestionStore.from_url(settings.sqlalchemy_url())


store_factory: Callable[[Settings], IngestionStore] = default_store  # remplacé dans les tests


@task(name="GET /admin/clock", task_run_name="recaler l'horloge sur le {watermark}", cache_policy=NO_CACHE)
def catch_up_clock(clock: SimulationClock, watermark: date) -> date | None:
    """Remet l'horloge de la source au watermark si elle est en retard (source redémarrée)."""
    behind = clock.catch_up(watermark)
    if behind is not None:
        get_run_logger().warning("horloge de la source en retard (redémarrée ?) : %s → %s, date jusqu'à laquelle "
                                 "la base est ingérée", behind, watermark)  # fmt: skip
    return behind


@task(name="POST /admin/clock/advance", task_run_name="avancer de {days} jour(s)", cache_policy=NO_CACHE)
def advance_clock(clock: SimulationClock, days: int) -> ClockStep:
    """Fait avancer le temps simulé et dit ce qui vient de se terminer."""
    step = clock.advance(days)
    get_run_logger().info(step.summary())
    return step


@flow(name=FLOW_NAME)
def simulation_flow(days: int | None = None, config_path: str | None = None) -> SimulationReport:
    """Un pas de simulation : avancer l'horloge, puis ingérer ce qui vient de se terminer."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = ProjectConfig.from_yaml(path)
    days = days or config.orchestration.simulation_days_per_run
    logger = get_run_logger()
    report = SimulationReport(days=days)
    settings = Settings()
    clock = clock_factory(settings, config)
    try:
        try:
            watermark = store_factory(settings).watermark(default=config.simulation.bootstrap_start)
            report.caught_up = catch_up_clock(clock, watermark)
        except (SQLAlchemyError, SourceUnavailableError) as exc:  # l'avance ou l'ingestion diront la panne
            logger.warning("horloge non recalée : %s", exc)
        step = advance_clock(clock, days)
        report.previous, report.now = step.previous, step.now
        report.new_matches, report.tournaments = step.new_matches, step.tournaments
    except ReplayFinishedError as exc:
        report.finished, report.detail = True, str(exc)
        logger.warning(report.summary())
        return report
    except SourceUnavailableError as exc:
        report.detail = f"horloge non avancée ({exc})"
        logger.warning("%s : l'ingestion constatera la panne", report.detail)
    finally:
        clock.close()
        name_run(report)
    state = ingest_flow(str(path), return_state=True)
    report.ingestion = state.name or state.type.value
    result: Any = state.result(raise_on_failure=False)
    if isinstance(result, IngestionReport):
        report.ingestion_summary = result.summary()
    elif isinstance(result, BaseException):
        report.ingestion_summary = f"{type(result).__name__} : {result}"[:300]
    report.feedback = _outcome(feedback_flow(return_state=True))
    if report.ingestion == DEGRADED_STATE:
        report.scoring = "sauté (source injoignable)"
    else:
        report.scoring = _outcome(scoring_flow(config_path=str(path), return_state=True))
    logger.info(report.summary())
    return report


def run_name(report: SimulationReport) -> str:
    """Le nom du run dans Prefect : la semaine simulée (`2017-01-07 → 2017-01-14`), lisible comme une frise."""
    if report.finished:
        return "fin du rejeu"
    if report.now is None:
        return "horloge non avancée"
    return f"{report.previous} → {report.now}" + (" (horloge recalée)" if report.caught_up else "")


def name_run(report: SimulationReport) -> None:
    """Renomme le run en cours (le nom au hasard de Prefect ne dit rien) ; sans effet hors d'un run, jamais bloquant."""
    run_id = flow_run.get_id()
    if run_id is None:
        return
    try:
        with get_client(sync_client=True) as client:
            client.update_flow_run(UUID(run_id), name=run_name(report))
    except (PrefectException, httpx.HTTPError, OSError) as exc:  # le pas compte plus que son nom
        get_run_logger().warning("run non renommé : %s", exc)


def _outcome(state: Any) -> str:  # noqa: ANN401
    """Bilan d'un sous-flow, ou son état s'il a échoué (il porte alors sa propre alerte)."""
    result = state.result(raise_on_failure=False)
    if isinstance(result, BaseException):
        return f"{state.name} ({type(result).__name__})"
    return str(result.summary()) if hasattr(result, "summary") else str(state.name)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m tennis_mlops.flows.simulation [--days 7]` : un pas, sans planning."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.flows.simulation", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--days", type=int, default=None, help="jours simulés (défaut : configs/project.yaml)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    report = simulation_flow(args.days, str(args.config))
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
