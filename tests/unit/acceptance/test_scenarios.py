from datetime import date
from typing import Any

import pytest

from tennis_mlops.acceptance import scenarios as sc
from tennis_mlops.acceptance.report import Recorder, ScenarioFailedError
from tennis_mlops.acceptance.scenarios import Context, FlowOutcome, Scenario, settle
from tennis_mlops.acceptance.stack import TaskInfo, WaitTimeoutError, now
from tennis_mlops.flows.simulation import SimulationReport
from tests.unit.acceptance.fakes import FakeActions, FakeServices, FakeStack


def play(scenario: Scenario, stack: FakeStack, actions: FakeActions | None = None) -> tuple[Recorder, FakeActions]:
    actions = actions or FakeActions(stack)
    rec = Recorder()
    scenario.run(Context(stack, FakeServices(stack), actions, rec))
    return rec, actions


def texts(rec: Recorder) -> str:
    return "\n".join(step.text for step in rec.steps)


def alert(kind: str, flow_run_id: str | None = None, delivery: str = "silenced") -> dict[str, Any]:
    return {"kind": kind, "title": f"carte {kind}", "delivery": delivery, "at": now().isoformat(),
            "flow_run_id": flow_run_id}  # fmt: skip


# --------------------------------------------------------------------------- S1


def nominal_world(gate: str = "passed") -> tuple[FakeStack, FakeActions]:
    stack = FakeStack()
    stack.sql = {
        "SELECT lot_id, status FROM ingestion_lots": [{"lot_id": "2015-0301", "status": "valid"}],
        "SELECT count(*) FROM predictions": 14,
        "SELECT count(*) FROM feedbacks": 9,
        "SELECT count(*) FROM prediction_outcomes": 9,
    }
    first = SimulationReport(days=154, previous=date(2016, 1, 1), now=date(2016, 6, 3), new_matches=262,
                             ingestion="Completed", ingestion_summary="ok : 12 lot(s) stocké(s)")  # fmt: skip
    second = SimulationReport(days=7, previous=date(2015, 2, 5), now=date(2015, 2, 12), feedback="9 résultat(s)")
    actions = FakeActions(stack, simulations=[FlowOutcome("sim-1", "Completed", False, first),
                                              FlowOutcome("sim-2", "Completed", False, second)])  # fmt: skip

    def chain(acts: FakeActions, days: int) -> None:
        if days != 154:
            return
        stack.event("training_requested", title="Réentraînement demandé : 262 nouveaux matchs")
        stack.training()
        stack.tags["2"] = {"gate_status": gate, "gate_reason": "log loss 0,601 ≤ 0,603"}
        if gate == "passed":
            stack.aliases.update(production="2", previous="1")
            stack.served = "2"

    actions.on_simulate = chain
    return stack, actions


def test_s1_follows_the_whole_chain() -> None:
    stack, actions = nominal_world()
    rec, _ = play(sc.S1Nominal(), stack, actions)
    out = texts(rec)
    for expected in ("production avant : v1", "Great Expectations : 1 lot(s) validé(s)", "réentraînement demandé",
                     "nouvelle version enregistrée : v2", "alias production déplacé : v1 → v2",
                     "l'API répond avec v2", "71.0%", "scoring : 14", "feedbacks enregistrés : 9"):  # fmt: skip
        assert expected in out
    assert actions.calls == ["simulate 154", "simulate 7"]
    assert [link.label for link in rec.links] == [
        "Prefect : pas de simulation",
        "Prefect : entraînement",
        "MLflow : v2",
    ]


def test_s1_fails_when_the_new_version_is_refused() -> None:
    stack, actions = nominal_world(gate="failed")
    with pytest.raises(ScenarioFailedError, match="règle de mise en service : failed"):
        play(sc.S1Nominal(), stack, actions)


# --------------------------------------------------------------------------- S2, S5 : quarantaine


def quarantine_world(mode: str, check: dict[str, Any]) -> tuple[FakeStack, FakeActions]:
    stack = FakeStack()
    stack.sql = {"SELECT count(*) FROM matches": 0,
                 "SELECT lot_id FROM ingestion_lots WHERE status = 'valid'": [{"lot_id": "2015-0400"}]}  # fmt: skip
    actions = FakeActions(
        stack, ingests=[FlowOutcome("ing-1", "Failed", True), FlowOutcome("ing-2", "Completed", False)]
    )

    def refuse(acts: FakeActions) -> None:
        if stack.chaos_mode == mode:
            stack.event("lot_quarantined", flow_run_id="ing-1", lot_id="2015-0400", payload={"failed_checks": [check]})
            stack.alert_list.insert(0, alert("quarantaine", "ing-1"))

    actions.on_ingest = refuse
    return stack, actions


def test_s2_a_corrupt_lot_is_blocked_and_then_recovered() -> None:
    stack, actions = quarantine_world("corrupt", {"expectation": "expect_tennis_score_to_be_valid", "target": "score"})
    rec, _ = play(sc.S2Corrupt(), stack, actions)
    out = texts(rec)
    assert "expectations custom en échec : expect_tennis_score_to_be_valid" in out
    assert "aucun entraînement demandé ni lancé" in out and "carte quarantaine" in out
    assert "reprise : incident arrêté, les 1 lot(s) refusé(s) passent" in out
    assert actions.calls == ["ingest chaos=corrupt", "ingest chaos=off"] and stack.chaos_mode == "off"


def test_s2_fails_without_a_custom_expectation() -> None:
    stack, actions = quarantine_world("corrupt", {"expectation": "expect_column_values_to_be_in_set"})
    with pytest.raises(ScenarioFailedError, match="aucune expectation custom"):
        play(sc.S2Corrupt(), stack, actions)


def test_s5_the_schema_check_blocks_the_lot() -> None:
    stack, actions = quarantine_world("schema", {"expectation": "expect_column_to_exist", "target": "match_id"})
    rec, _ = play(sc.S5Schema(), stack, actions)
    assert "contrôle de schéma en échec : colonnes absentes match_id" in texts(rec)


# --------------------------------------------------------------------------- S3, S4 : la source en panne


def test_s3_retries_then_degraded() -> None:
    stack = FakeStack()
    stack.tasks = {"a": [TaskInfo("GET /tournaments since=2015-02-12", 3, "Completed", 3.1)],
                   "b": [TaskInfo("GET /tournaments since=2015-02-19", 2, "Completed", 12.4)],
                   "c": [TaskInfo("GET /tournaments since=2015-02-26", 4, "Failed", 7.2)]}  # fmt: skip
    outcomes = [FlowOutcome("a", "Completed", False), FlowOutcome("b", "Completed", False),
                FlowOutcome("c", "Dégradé", False)]  # fmt: skip
    actions = FakeActions(stack, ingests=outcomes)
    actions.on_ingest = lambda acts: stack.alert_list.insert(0, alert("degrade", "c")) if len(acts.calls) == 3 else None
    rec, _ = play(sc.S3Retries(), stack, actions)
    out = texts(rec)
    assert "joué 3 fois" in out and "réussite au 2e essai" in out and "4 essais (3 retries)" in out
    assert actions.calls == ["ingest chaos=503", "ingest chaos=slow", "ingest chaos=503"] and stack.chaos_mode == "off"


def test_s4_the_source_is_always_restarted_and_its_clock_restored() -> None:
    stack = FakeStack(now_day=date(2015, 3, 5), mark=date(2015, 3, 5))
    actions = FakeActions(stack, ingests=[FlowOutcome("s4", "Dégradé", False)])

    def down(acts: FakeActions) -> None:
        stack.event("ingestion_degraded", flow_run_id="s4")
        stack.alert_list.insert(0, alert("degrade", "s4"))

    actions.on_ingest = down
    rec, _ = play(sc.S4SourceDown(), stack, actions)
    out = texts(rec)
    assert "état « Dégradé » après 4 essais" in out and "watermark inchangé (2015-03-05)" in out
    assert "l'API sert toujours : données validées jusqu'au 2015-01-31" in out
    assert "horloge remise au 2015-03-05" in out and "mock_source" in stack.up


def test_s4_restarts_the_source_even_when_a_check_fails() -> None:
    stack = FakeStack(now_day=date(2015, 3, 5))
    actions = FakeActions(stack, ingests=[FlowOutcome("s4", "Completed", False)])  # pas dégradé : échec
    with pytest.raises(ScenarioFailedError):
        play(sc.S4SourceDown(), stack, actions)
    assert "mock_source" in stack.up and stack.now_day == date(2015, 3, 5)


# --------------------------------------------------------------------------- S6 → S9


def test_s6_a_weaker_model_is_registered_then_refused() -> None:
    rec, actions = play(sc.S6WeakModel(), FakeStack())
    out = texts(rec)
    assert "gate_status=failed, lifecycle=rejected" in out and "production inchangée (v1)" in out
    assert "alias staging" in out and actions.calls == ["train-weak"]


def test_s7_rollback_is_served_without_restarting_the_api() -> None:
    stack = FakeStack(aliases={"production": "2", "previous": "1"}, served="2",
                      tags={"1": {"lifecycle": "archived"}, "2": {"lifecycle": "production"}})  # fmt: skip
    rec, _ = play(sc.S7Rollback(), stack)
    out = texts(rec)
    assert "alias production : v2 → v1" in out and "v2 marquée rolled_back" in out
    assert "l'API sert v1 sans redémarrage (relecture de l'alias n° 4)" in out and "carte rollback" not in out
    assert "↩️ Rollback" in out


def test_s7_needs_a_previous_version() -> None:
    with pytest.raises(ScenarioFailedError, match="aucune version previous"):
        play(sc.S7Rollback(), FakeStack())


def test_s8_cache_and_422_then_mlflow_is_back() -> None:
    stack = FakeStack()
    stack.sql = {"SELECT status_code, error_type, message FROM errors": [
        {"status_code": 422, "error_type": "validation_error", "message": "surface : valeur inattendue"}]}  # fmt: skip
    rec, _ = play(sc.S8MlflowDown(), stack)
    out = texts(rec)
    assert "/ready : degraded — registre MLflow injoignable" in out and "servie depuis le cache : v1" in out
    assert "HTTP 422 (bad-1)" in out and "422 tracé dans errors : validation_error" in out
    assert "mlflow" in stack.up and "de nouveau « ready »" in out


def test_s9_reports_the_ui_smoke_steps_or_its_failure() -> None:
    rec, _ = play(sc.S9Interface(), FakeStack())
    assert [s.text for s in rec.steps] == ["interface en ligne", "prédiction affichée", "tracée"]

    class Broken(FakeActions):
        def ui_smoke(self, ui_url: str, settings: Any) -> list[str]:
            raise AssertionError("aucune ligne predictions pour request_id x")

    stack = FakeStack()
    with pytest.raises(ScenarioFailedError, match="aucune ligne predictions"):
        play(sc.S9Interface(), stack, Broken(stack))


# --------------------------------------------------------------------------- settle


def test_settle_waits_for_requested_trainings_and_names_a_missing_automation(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = FakeStack()
    start = now()
    stack.event("training_requested")
    stack.training(state="COMPLETED")
    assert [r.state_type for r in settle(stack, start)] == ["COMPLETED"]

    monkeypatch.setattr(sc, "AUTOMATION_GRACE_S", -1)
    stack.event("training_requested")  # une 2e demande, jamais suivie d'un run
    with pytest.raises(WaitTimeoutError, match="2 entraînement\\(s\\) demandé\\(s\\), 1 lancé"):
        settle(stack, start, timeout_s=30)
