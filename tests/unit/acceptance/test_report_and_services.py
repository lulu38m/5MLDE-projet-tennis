import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tennis_mlops.acceptance.report import (
    CampaignResult,
    Link,
    Recorder,
    ScenarioFailedError,
    ScenarioResult,
    Status,
    Step,
)
from tennis_mlops.acceptance.services import ComposeServices, ServiceControlError

T0 = datetime(2026, 11, 18, 9, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 18, 9, 14, tzinfo=UTC)


def campaign(*statuses: Status) -> CampaignResult:
    scenarios = [ScenarioResult(f"S{i + 1}", f"titre {i + 1}", "attendu", s, 12.0) for i, s in enumerate(statuses)]
    return CampaignResult(T0, T1, {"machine": "mac"}, scenarios)


def test_the_recorder_stops_at_the_first_false_check() -> None:
    rec = Recorder()
    rec.step("action faite")
    rec.expect(True, "vrai")
    with pytest.raises(ScenarioFailedError, match="raison précise"):
        rec.expect(0, "texte si vrai", "raison précise")
    rec.link("Prefect", "http://127.0.0.1:4201/runs/flow-run/x")
    assert rec.steps == [Step(True, "action faite"), Step(True, "vrai"), Step(False, "raison précise")]
    assert rec.links == [Link("Prefect", "http://127.0.0.1:4201/runs/flow-run/x")]


def test_green_only_when_every_played_scenario_passed() -> None:
    assert campaign(Status.PASSED, Status.SKIPPED).passed
    assert not campaign(Status.PASSED, Status.FAILED).passed
    assert not campaign(Status.PASSED, Status.ERROR).passed
    assert not campaign(Status.SKIPPED).passed  # rien joué : pas vert
    summary = campaign(Status.PASSED, Status.FAILED, Status.SKIPPED).summary()
    assert summary.startswith("ROUGE : 1 réussi(s), 1 échoué(s), 0 en erreur, 1 ignoré(s)") and "14.0 min" in summary


def test_report_files_are_self_contained_and_escaped(tmp_path: Path) -> None:
    result = campaign(Status.PASSED, Status.FAILED, Status.SKIPPED)
    result.scenarios[1].steps = [Step(True, "lot <b>refusé</b>"), Step(False, "alerte absente")]
    result.scenarios[1].links = [Link("MLflow v2", "http://127.0.0.1:7282/#/models/atp-winner/versions/2")]
    result.scenarios[1].error = "WaitTimeoutError : trop long"
    page = result.write(tmp_path / "acceptance")
    text = page.read_text(encoding="utf-8")
    assert "lot &lt;b&gt;refusé&lt;/b&gt;" in text and "<b>refusé</b>" not in text
    assert "<script" not in text and "http" not in text.split("<body>")[0]  # aucune ressource externe
    assert 'id="S1"' in text and 'id="S3"' not in text  # un scénario ignoré n'a pas de section
    assert "versions/2" in text and "WaitTimeoutError" in text
    data = json.loads((tmp_path / "acceptance" / "rapport.json").read_text(encoding="utf-8"))
    assert data["passed"] is False and [s["status"] for s in data["scenarios"]] == ["passed", "failed", "skipped"]


class Runner:
    def __init__(self, code: int = 0, error: Exception | None = None) -> None:
        self.code, self.error = code, error
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(command), kwargs["cwd"]))
        if self.error:
            raise self.error
        return subprocess.CompletedProcess(command, self.code, "", "no such service: mlfow")


def test_compose_services_stop_and_start_in_the_project_folder(tmp_path: Path) -> None:
    runner = Runner()
    services = ComposeServices(tmp_path, runner)
    services.stop("mock_source")
    services.start("mock_source")
    assert runner.calls == [(["docker", "compose", "stop", "mock_source"], tmp_path),
                            (["docker", "compose", "start", "mock_source"], tmp_path)]  # fmt: skip


def test_compose_errors_are_reported_with_their_cause(tmp_path: Path) -> None:
    with pytest.raises(ServiceControlError, match="code 1 — no such service"):
        ComposeServices(tmp_path, Runner(code=1)).stop("mlfow")
    with pytest.raises(ServiceControlError, match="FileNotFoundError"):
        ComposeServices(tmp_path, Runner(error=FileNotFoundError("docker"))).start("mlflow")
