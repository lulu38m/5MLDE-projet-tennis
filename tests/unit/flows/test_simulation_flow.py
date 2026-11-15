"""Flow de simulation : un pas d'horloge, ingestion, feedback, scoring ; un incident = une alerte ; fin du rejeu."""

from datetime import date
from typing import Any

import httpx
import pytest
from prefect import flow

from mock_source.chaos import ChaosMode, ChaosSettings
from tennis_mlops.config import ProjectConfig, Settings
from tennis_mlops.flows import ingest, simulation
from tennis_mlops.flows.simulation import SimulationReport
from tennis_mlops.orchestration.clock import SimulationClock
from tests.unit.flows.conftest import FlowWorld


@pytest.fixture
def sim(world: FlowWorld, monkeypatch: pytest.MonkeyPatch) -> FlowWorld:
    def clock(_: Settings, __: ProjectConfig) -> SimulationClock:
        return SimulationClock("", 5, http=world.env.client)

    monkeypatch.setattr(simulation, "clock_factory", clock)
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


def test_each_step_advances_the_clock_then_ingests(sim: FlowWorld) -> None:
    sim.env.at(date(2019, 1, 6))
    state, report = step(sim, 8)  # 06/01 → 14/01 : l'Alpha Open (finale le 13) vient de se terminer
    assert state.is_completed()
    assert (report.previous, report.now, report.tournaments) == (date(2019, 1, 6), date(2019, 1, 14), ["2019-0001"])
    assert report.new_matches == 7 and report.ingestion == "Completed"
    assert "1 lot(s) stocké(s)" in report.ingestion_summary and sim.store.count_matches() == 7
    assert (report.feedback, report.scoring) == ("aucun résultat à remonter", "3 match(s) prédit(s)")


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
    assert report.ingestion == "Completed"  # l'ingestion a tourné (ici la source répond encore)


def test_end_of_replay(sim: FlowWorld) -> None:
    sim.env.at(date(2020, 12, 31))
    state, report = step(sim, 7)
    assert state.is_completed() and report.finished and report.ingestion == "non lancée"
    assert report.summary().startswith("fin du rejeu")
