from datetime import date
from typing import Any

from prefect import State
from prefect.states import Completed, Failed

from mock_source.chaos import ChaosMode, ChaosSettings
from tennis_mlops.data.errors import SourceRequestError, SourceUnavailableError
from tennis_mlops.flows import ingest
from tennis_mlops.flows.ingest import IngestionReport, IngestionStatus
from tennis_mlops.ingestion.store import LotStatus

from .conftest import FlowWorld

BOOTSTRAP = "bootstrap_2019-01-01_2019-02-01"


def run(world: FlowWorld, day: date) -> tuple[State[Any], IngestionReport | None]:
    world.env.at(day)
    state = ingest.ingest_flow(str(world.config_path), return_state=True)
    result = state.result(raise_on_failure=False)
    return state, result if isinstance(result, IngestionReport) else None


def watermark(world: FlowWorld) -> date:
    return world.store.watermark(default=date(2000, 1, 1))


def test_nominal_bootstrap_then_new_tournaments(world: FlowWorld) -> None:
    state, report = run(world, date(2019, 2, 1))
    assert state.is_completed() and report is not None
    assert (report.status, report.lots_valid, report.rows_inserted) == (IngestionStatus.OK, [BOOTSTRAP], 7)
    assert watermark(world) == date(2019, 2, 1)

    state, report = run(world, date(2019, 3, 10))
    assert report is not None and (report.lots_valid, report.rows_inserted) == (["2019-D001"], 4)
    assert watermark(world) == date(2019, 3, 10)
    assert (world.state_dir / "lots" / "2019-D001" / "payload.json").is_file()

    state, report = run(world, date(2019, 3, 10))
    assert report is not None and report.status is IngestionStatus.UP_TO_DATE
    assert world.store.count_matches() == 11


def test_invalid_lot_is_quarantined_fails_the_flow_and_is_retried_later(world: FlowWorld) -> None:
    run(world, date(2019, 2, 1))
    world.env.chaos.configure(ChaosSettings(mode=ChaosMode.SCHEMA))
    state, _ = run(world, date(2019, 3, 10))
    assert state.is_failed()
    assert "2019-D001" in str(state.message)
    trace = world.store.lot("2019-D001")
    assert trace is not None and trace["status"] == LotStatus.QUARANTINED
    assert (world.state_dir / "quarantine" / "2019-D001" / "validation.json").is_file()
    assert world.store.count_matches() == 7  # rien du lot refusé n'est stocké
    assert watermark(world) == date(2019, 3, 3)  # arrêté à la fin du lot refusé : il sera retenté

    world.env.chaos.configure(ChaosSettings(mode=ChaosMode.OFF))
    state, report = run(world, date(2019, 3, 10))
    assert state.is_completed() and report is not None and report.lots_valid == ["2019-D001"]
    trace = world.store.lot("2019-D001")
    assert trace is not None and trace["status"] == LotStatus.VALID
    assert watermark(world) == date(2019, 3, 10)


def test_transient_outage_is_absorbed_by_retries(world: FlowWorld) -> None:
    world.env.chaos.configure(ChaosSettings(mode=ChaosMode.UNAVAILABLE, failures=2))
    state, report = run(world, date(2019, 2, 1))
    assert state.is_completed() and report is not None and report.status is IngestionStatus.OK


def test_persistent_outage_ends_degraded_without_losing_anything(world: FlowWorld) -> None:
    run(world, date(2019, 2, 1))
    world.env.chaos.configure(ChaosSettings(mode=ChaosMode.UNAVAILABLE))
    state, report = run(world, date(2019, 3, 10))
    assert state.is_completed() and state.name == ingest.DEGRADED_STATE
    assert report is not None and report.status is IngestionStatus.DEGRADED and "503" in str(report.detail)
    assert watermark(world) == date(2019, 2, 1)
    assert world.store.count_matches() == 7


def test_command_line_exit_codes(world: FlowWorld) -> None:
    world.env.at(date(2019, 2, 1))
    assert ingest.main(["--config", str(world.config_path)]) == ingest.EXIT_OK
    world.env.at(date(2019, 3, 10))
    world.env.chaos.configure(ChaosSettings(mode=ChaosMode.SCHEMA))
    assert ingest.main(["--config", str(world.config_path)]) == ingest.EXIT_QUARANTINE
    world.env.chaos.configure(ChaosSettings(mode=ChaosMode.UNAVAILABLE))
    assert ingest.main(["--config", str(world.config_path)]) == ingest.EXIT_DEGRADED


def test_only_outages_are_retried() -> None:
    task: Any = None
    run_: Any = None
    assert ingest.retry_on_outage(task, run_, Failed(data=SourceUnavailableError("503")))
    assert not ingest.retry_on_outage(task, run_, Failed(data=SourceRequestError("404")))
    assert not ingest.retry_on_outage(task, run_, Completed(data=1))


def test_report_summary_is_readable() -> None:
    report = IngestionReport(
        status=IngestionStatus.OK,
        simulated_now=date(2019, 3, 10),
        watermark_before=date(2019, 2, 1),
        watermark_after=date(2019, 3, 10),
        lots_valid=["2019-D001"],
        rows_inserted=4,
    )
    assert report.summary() == (
        "ok : 1 lot(s) stocké(s), 0 en quarantaine, 4 match(s) ajouté(s), watermark 2019-02-01 → 2019-03-10"
    )
