"""Flow de simulation : un pas d'horloge, ingestion, feedback, scoring ; un incident = une alerte ; fin du rejeu."""

from datetime import date, timedelta
from typing import Any

import httpx
import pytest
from prefect import flow, get_client

from mock_source.chaos import ChaosMode, ChaosSettings
from tennis_mlops.config import ProjectConfig, Settings
from tennis_mlops.data.errors import SourceUnavailableError
from tennis_mlops.flows import ingest, simulation
from tennis_mlops.flows.simulation import SimulationReport
from tennis_mlops.orchestration.clock import SimulationClock
from tests.unit.flows.conftest import FlowWorld


@pytest.fixture
def sim(world: FlowWorld, monkeypatch: pytest.MonkeyPatch) -> FlowWorld:
    def clock(_: Settings, __: ProjectConfig) -> SimulationClock:
        return SimulationClock("", 5, http=world.env.client)

    monkeypatch.setattr(simulation, "clock_factory", clock)
    monkeypatch.setattr(simulation, "store_factory", lambda _: world.store)
    monkeypatch.setattr(ingest, "training_event_emitter", lambda **_: None)
    monkeypatch.setattr(simulation, "feedback_flow", fake_feedback)  # scoring et feedback : leurs propres tests
    monkeypatch.setattr(simulation, "scoring_flow", fake_scoring)
    return world


class Done:
    def __init__(self, text: str) -> None:
        self.text = text

    def summary(self) -> str:
        return self.text


@flow(name="feedback-atp")
def fake_feedback() -> Done:
    return Done("aucun résultat à remonter")


@flow(name="scoring-atp")
def fake_scoring(config_path: str | None = None) -> Done:
    return Done("3 match(s) prédit(s)")


def step(world: FlowWorld, days: int) -> tuple[Any, SimulationReport]:
    state = simulation.simulation_flow(days, str(world.config_path), return_state=True)
    return state, state.result()


def run_name_of(state: Any) -> str:
    with get_client(sync_client=True) as client:
        return client.read_flow_run(state.state_details.flow_run_id).name


def test_each_step_advances_the_clock_then_ingests(sim: FlowWorld) -> None:
    sim.env.at(date(2019, 1, 6))
    state, report = step(sim, 8)  # 06/01 → 14/01 : l'Alpha Open (finale le 13) vient de se terminer
    assert state.is_completed()
    assert (report.previous, report.now, report.tournaments) == (date(2019, 1, 6), date(2019, 1, 14), ["2019-0001"])
    assert report.new_matches == 7 and report.ingestion == "Completed"
    assert "1 lot(s) stocké(s)" in report.ingestion_summary and sim.store.count_matches() == 7
    assert (report.feedback, report.scoring) == ("aucun résultat à remonter", "3 match(s) prédit(s)")
    assert run_name_of(state) == "2019-01-06 → 2019-01-14"  # la semaine simulée, plutôt qu'un nom au hasard


def test_a_restarted_source_is_put_back_on_the_watermark(sim: FlowWorld) -> None:
    """mock_source redémarré : son horloge repart du début du rejeu alors que la base est déjà ingérée."""
    sim.env.at(date(2019, 1, 6))
    step(sim, 8)  # ingère l'Alpha Open, watermark au 14/01
    watermark = sim.store.watermark(default=date(2015, 1, 1))
    sim.env.at(date(2018, 12, 15))  # redémarrage : l'horloge recule
    state, report = step(sim, 7)
    assert state.is_completed()
    assert report.caught_up == date(2018, 12, 15)
    assert (report.previous, report.now) == (watermark, watermark + timedelta(days=7))
    assert report.summary().startswith(f"pas de 7 jour(s) : horloge recalée depuis 2018-12-15, {watermark}")
    assert run_name_of(state).endswith("(horloge recalée)")
    _, again = step(sim, 7)
    assert again.caught_up is None  # à l'heure : rien à recaler


def test_an_ingestion_incident_does_not_fail_the_simulation(sim: FlowWorld) -> None:
    sim.env.at(date(2019, 2, 25))
    sim.env.chaos.configure(ChaosSettings(mode=ChaosMode.SCHEMA))
    state, report = step(sim, 14)
    assert state.is_completed()  # la simulation réussit : seul le sous-flow d'ingestion porte l'alerte
    assert report.ingestion == "Failed" and "DataQualityError" in report.ingestion_summary

    sim.env.chaos.configure(ChaosSettings(mode=ChaosMode.UNAVAILABLE, failures=50))
    _, degraded = step(sim, 1)
    assert degraded.ingestion == "Dégradé" and degraded.now == date(2019, 3, 12)  # l'horloge, elle, a avancé
    assert degraded.scoring == "sauté (source injoignable)"  # la panne a déjà son alerte


def test_unreachable_clock_still_runs_the_ingestion(sim: FlowWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refusé")

    unreachable = httpx.Client(base_url="http://mock", transport=httpx.MockTransport(broken))
    monkeypatch.setattr(simulation, "clock_factory", lambda *_: SimulationClock("", 5, http=unreachable))
    sim.env.at(date(2019, 1, 14))
    state, report = step(sim, 7)
    assert state.is_completed() and report.now is None and "horloge non avancée" in report.detail
    assert run_name_of(state) == "horloge non avancée"
    assert report.ingestion == "Completed"  # l'ingestion a tourné (ici la source répond encore)


def test_end_of_replay(sim: FlowWorld) -> None:
    sim.env.at(date(2020, 12, 31))
    state, report = step(sim, 7)
    assert state.is_completed() and report.finished and report.ingestion == "non lancée"
    assert run_name_of(state) == "fin du rejeu"
    assert report.summary().startswith("fin du rejeu")


def test_catch_up_reports_source_failures() -> None:
    def answer(status: int) -> SimulationClock:
        def handle(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"now": "2016-01-01"})
            return httpx.Response(status, text="hors de la fenêtre")

        return SimulationClock("", 5, http=httpx.Client(base_url="http://mock", transport=httpx.MockTransport(handle)))

    assert answer(200).catch_up(date(2015, 6, 1)) is None  # déjà en avance : rien envoyé
    assert answer(200).catch_up(date(2017, 1, 1)) == date(2016, 1, 1)
    with pytest.raises(SourceUnavailableError, match="HTTP 422"):
        answer(422).catch_up(date(2017, 1, 1))

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refusé", request=request)

    down = SimulationClock("", 5, http=httpx.Client(base_url="http://mock", transport=httpx.MockTransport(refused)))
    with pytest.raises(SourceUnavailableError, match="injoignable"):
        down.catch_up(date(2017, 1, 1))
