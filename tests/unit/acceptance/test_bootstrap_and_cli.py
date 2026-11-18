from datetime import date

import pytest

from tennis_mlops.acceptance import __main__ as cli
from tennis_mlops.acceptance.bootstrap import BootstrapError, bootstrap
from tennis_mlops.acceptance.scenarios import FlowOutcome
from tennis_mlops.acceptance.urls import lines
from tennis_mlops.config import Settings
from tennis_mlops.flows.ingest import IngestionReport, IngestionStatus
from tests.unit.acceptance.fakes import FakeActions, FakeStack

REPORT = IngestionReport(status=IngestionStatus.OK, simulated_now=date(2016, 1, 1),
                         watermark_before=date(2010, 1, 1), watermark_after=date(2016, 1, 1),
                         lots_valid=["bootstrap-2010-2015"], rows_inserted=14899)  # fmt: skip


def empty_stack() -> FakeStack:
    return FakeStack(aliases={}, tags={}, served=None, mark=date(2010, 1, 1))


def test_a_new_stack_gets_its_first_model_through_the_chain() -> None:
    stack = empty_stack()
    actions = FakeActions(stack, ingests=[FlowOutcome("boot", "Completed", False, REPORT)])

    def chain(acts: FakeActions) -> None:  # l'ingestion demande l'entraînement, l'automation le lance
        stack.event("training_requested")
        stack.training()
        stack.tags["1"] = {"test.log_loss": "0.603", "data_window_end": "2014-12-31"}
        stack.aliases["production"], stack.served = "1", "1"

    actions.on_ingest = chain
    log: list[str] = []
    assert bootstrap(stack, actions, log.append) == "1"
    assert "14899 match(s) ajouté(s)" in "\n".join(log)
    assert log[-1] == "modèle en production : v1 (log loss test 0.603, données jusqu'au 2014-12-31), servi par l'API"


def test_bootstrap_is_idempotent_and_relaunches_a_missing_training() -> None:
    ready = FakeStack()
    actions = FakeActions(ready)
    assert bootstrap(ready, actions, lambda _: None) == "1" and actions.calls == []  # rien à refaire

    ingested = empty_stack()
    ingested.mark = date(2016, 1, 1)  # historique déjà là, mais pas de modèle (entraînement raté)
    requested: list[str] = []

    def relaunch() -> None:
        requested.append("run_deployment")
        ingested.training()
        ingested.tags["1"] = {}
        ingested.aliases["production"], ingested.served = "1", "1"

    assert bootstrap(ingested, FakeActions(ingested), lambda _: None, relaunch) == "1"
    assert requested == ["run_deployment"]


def test_bootstrap_says_why_there_is_no_model() -> None:
    stack = empty_stack()
    refused = FakeActions(stack, ingests=[FlowOutcome("boot", "Failed", True, ValueError("lot refusé"))])
    with pytest.raises(BootstrapError, match="ingestion initiale : Failed"):
        bootstrap(stack, refused, lambda _: None)

    stack = empty_stack()
    actions = FakeActions(stack, ingests=[FlowOutcome("boot", "Completed", False, REPORT)])

    def rejected(acts: FakeActions) -> None:
        stack.event("training_requested")
        stack.training()
        stack.tags["1"] = {"gate_reason": "ne bat pas l'Elo pur"}

    actions.on_ingest = rejected
    with pytest.raises(BootstrapError, match="ne bat pas l'Elo pur"):
        bootstrap(stack, actions, lambda _: None)


def test_urls_show_access_and_whether_each_interface_answers() -> None:
    settings = Settings(_env_file=None)
    out = lines(settings, get=lambda url: 200 if "8501" in url else 0)
    assert out[0].startswith("● Interface de test") and "http://127.0.0.1:8501" in out[0]
    grafana = next(line for line in out if "Grafana" in line)
    assert grafana.startswith("○") and "admin / admin-local" in grafana
    assert any("tennis / tennis-local (make db)" in line for line in out)


def test_cli_rejects_unknown_scenarios(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--only", "S1,S12"]) == cli.EXIT_NOT_READY
    assert "scénarios inconnus : S12" in capsys.readouterr().err
