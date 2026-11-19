import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tennis_mlops.cli import Cli, main


class Recorder:
    """Faux `subprocess.run` : note chaque commande, renvoie le code prévu."""

    def __init__(self, codes: dict[str, int] | None = None, missing: tuple[str, ...] = ()) -> None:
        self.codes = codes or {}
        self.missing = missing
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] in self.missing:
            raise FileNotFoundError(command[0])
        self.calls.append((command, kwargs.get("env")))
        key = " ".join(command[2:] if command[0] == sys.executable else command)
        code = next((c for prefix, c in self.codes.items() if key.startswith(prefix)), 0)
        return subprocess.CompletedProcess(command, code, "abc123\n" if command[:2] == ["git", "rev-parse"] else "", "")

    def lines(self) -> list[str]:
        shown = []
        for command, _ in self.calls:
            shown.append(" ".join(["python", *command[1:]] if command[0] == sys.executable else command))
        return shown


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("GIT_COMMIT", "PREFECT_API_URL", "HOST_UID", "HOST_GID"):  # exportées par le Makefile (make check)
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env.example").write_text("MOCK_START_DATE=2016-01-01\n", encoding="utf-8")
    return tmp_path


def test_up_creates_env_checks_docker_then_data_stack_and_first_model(project: Path) -> None:
    run, log = Recorder(), list[str]()
    assert Cli(project, run, log.append).up() == 0
    assert (project / ".env").read_text(encoding="utf-8") == "MOCK_START_DATE=2016-01-01\n"
    assert (project / "state").is_dir()
    assert [line for line in run.lines() if not line.startswith("git")] == [
        "docker info --format {{.ServerVersion}}",
        "python -m tennis_mlops.data download",
        "docker compose up -d --build --wait",
        "python -m tennis_mlops.acceptance bootstrap",
    ]
    env = next(e for c, e in run.calls if c[:3] == ["docker", "compose", "up"])
    assert env is not None and env["GIT_COMMIT"] == "abc123" and env["PREFECT_API_URL"].endswith(":4201/api")
    if hasattr(os, "getuid"):
        assert env["HOST_UID"] == str(os.getuid())
    assert ".env créé" in log[0]


def test_up_keeps_an_existing_env_and_stops_at_the_first_failing_step(project: Path) -> None:
    (project / ".env").write_text("TENNIS_TEAMS_WEBHOOK_URL=secret\n", encoding="utf-8")
    run, log = Recorder({"docker compose up": 1}), list[str]()
    assert Cli(project, run, log.append).up() == 1
    assert (project / ".env").read_text(encoding="utf-8") == "TENNIS_TEAMS_WEBHOOK_URL=secret\n"
    assert "python -m tennis_mlops.acceptance bootstrap" not in run.lines() and "arrêt" in log[-1]


def test_up_says_what_to_do_when_docker_is_missing_or_closed(project: Path) -> None:
    log: list[str] = []
    assert Cli(project, Recorder(missing=("docker",)), log.append).up() == 1
    assert "installer Docker Desktop" in log[-1]
    assert Cli(project, Recorder({"docker info": 1}), log.append).up() == 1
    assert "ouvrir Docker Desktop" in log[-1]


def test_reset_removes_volumes_and_local_state_but_keeps_downloaded_data(project: Path) -> None:
    for name in ("state/gx/x", "state/lots/2016-0301", "state/acceptance", "data/raw"):
        (project / name).mkdir(parents=True)
    run = Recorder()
    assert Cli(project, run, lambda _: None).reset() == 0
    assert "docker compose down --volumes --remove-orphans" in run.lines()
    assert not (project / "state" / "gx").exists() and not (project / "state" / "acceptance").exists()
    assert (project / "data" / "raw").is_dir() and (project / "state").is_dir()


def test_commands_forward_to_the_project_modules(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = Recorder()
    cli = Cli(project, run, print)
    assert main(["scenarios", "--only", "S2,S5", "--silence"], cli) == 0
    assert main(["urls"], cli) == 0 and main(["demo"], cli) == 0 and main(["demo-stop"], cli) == 0
    assert main(["down"], cli) == 0
    assert [line for line in run.lines() if not line.startswith("git")] == [
        "python -m tennis_mlops.acceptance run --only S2,S5 --silence",
        "python -m tennis_mlops.acceptance urls",
        "python -m tennis_mlops.orchestration start",
        "python -m tennis_mlops.orchestration stop",
        "docker compose down",
    ]
    assert main(["report"], cli) == 1 and "uv run tennis scenarios" in capsys.readouterr().out


def test_project_prefect_url_wins_over_a_global_one(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREFECT_API_URL", "http://localhost:4200/api")  # laissé par les TP dans l'environnement
    assert Cli(project, Recorder(), print).env()["PREFECT_API_URL"].endswith(":4201/api")
