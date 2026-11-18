from datetime import date

import pytest

from tennis_mlops.acceptance.report import Status
from tennis_mlops.acceptance.runner import Campaign, NotReadyError
from tennis_mlops.acceptance.scenarios import Context, Scenario, scenario_ids
from tennis_mlops.acceptance.stack import WaitTimeoutError
from tests.unit.acceptance.fakes import FakeActions, FakeServices, FakeStack


class Passes(Scenario):
    id, title, expected = "S1", "réussit", "tout va bien"

    def run(self, ctx: Context) -> None:
        ctx.rec.expect(True, "vérifié")


class Fails(Scenario):
    id, title, expected = "S2", "échoue", "une vérification fausse"

    def run(self, ctx: Context) -> None:
        ctx.stack.chaos("corrupt")
        ctx.rec.expect(False, "", "alerte absente")


class TimesOut(Scenario):
    id, title, expected = "S3", "trop long", "attente"

    def run(self, ctx: Context) -> None:
        ctx.services.stop("mock_source")  # le scénario s'interrompt sans relancer la source
        raise WaitTimeoutError("alerte « degrade » : pas au bout de 120 s")


class Crashes(Scenario):
    id, title, expected = "S4", "plante", "bug"

    def run(self, ctx: Context) -> None:
        raise ValueError("inattendu")


def campaign(stack: FakeStack, **kwargs: object) -> tuple[Campaign, list[str], FakeServices]:
    log: list[str] = []
    services = FakeServices(stack)
    scenarios = kwargs.pop("scenarios", (Passes, Fails, TimesOut, Crashes))
    c = Campaign(stack, services, FakeActions(stack), scenarios, log=log.append, **kwargs)  # type: ignore[arg-type]
    return c, log, services


def test_every_scenario_is_played_and_the_stack_is_put_back_in_order() -> None:
    stack = FakeStack(teams=True)
    c, log, services = campaign(stack, teams=False)
    result = c.run()
    assert [s.status for s in result.scenarios] == [Status.PASSED, Status.FAILED, Status.ERROR, Status.ERROR]
    assert result.scenarios[1].steps[-1].text == "alerte absente"
    assert "pas au bout de 120 s" in result.scenarios[2].error and "ValueError" in result.scenarios[3].error
    assert services.calls == ["stop mock_source", "start mock_source"]  # relancée après le scénario qui l'a arrêtée
    assert "clock=2016-01-01" in stack.calls  # horloge remise au watermark
    assert stack.chaos_mode == "off" and stack.calls[0] == "silence" and stack.calls[-1] == "unsilence"
    assert any("mode silence" in note for note in result.notes)
    assert not result.passed and any("▶ S2 · échoue" in line for line in log)


def test_only_and_teams() -> None:
    stack = FakeStack(paused=False)
    c, _, _ = campaign(stack, only=frozenset({"S1"}))  # par défaut : les cartes partent
    result = c.run()
    assert [s.status for s in result.scenarios] == [Status.PASSED, Status.SKIPPED, Status.SKIPPED, Status.SKIPPED]
    assert result.passed and "silence" not in stack.calls and "unsilence" not in stack.calls
    assert stack.paused and any("simulation planifiée mise en pause" in n for n in result.notes)
    assert any("cartes Teams envoyées" in n for n in result.notes)


def test_a_stack_that_is_down_or_already_used_is_refused_with_the_command_to_run() -> None:
    stack = FakeStack()
    stack.up.discard("prefect")
    with pytest.raises(NotReadyError, match="services injoignables : prefect"):
        campaign(stack)[0].run()
    assert "silence" not in stack.calls  # rien n'a été touché

    used = FakeStack(now_day=date(2017, 3, 4), tags={"1": {}, "2": {}, "3": {}})
    with pytest.raises(
        NotReadyError,
        match=r"3 versions .* horloge simulée au 2017-03-04 .*uv run tennis reset.*make reset && make up",
    ):
        campaign(used)[0].run()

    forced, _, _ = campaign(used, force=True, scenarios=(Passes,))
    result = forced.run()
    assert result.passed and any("--force" in note for note in result.notes)


def test_the_nine_scenarios_are_in_order() -> None:
    assert scenario_ids() == [f"S{i}" for i in range(1, 10)]
