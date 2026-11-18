"""Arrêter et relancer un service de la stack (S4 : `mock_source`, S8 : `mlflow`).

La stack est celle de `docker compose` ; un autre pilote peut être fourni (tests, stack hors Docker) tant qu'il sait
arrêter et relancer un service par son nom.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

COMPOSE_TIMEOUT_S = 180


class ServiceControlError(RuntimeError):
    """Le service n'a pas pu être arrêté ou relancé."""


class ServiceControl(Protocol):
    """Ce dont les scénarios ont besoin pour simuler une panne."""

    def stop(self, service: str) -> None:
        """Arrête le service (son conteneur reste là, ses données aussi)."""
        ...

    def start(self, service: str) -> None:
        """Relance un service arrêté."""
        ...


Runner = Callable[..., subprocess.CompletedProcess[str]]


class ComposeServices:
    """`docker compose stop|start <service>` dans le dossier du projet."""

    def __init__(self, project_dir: Path, runner: Runner = subprocess.run) -> None:
        self.project_dir = project_dir
        self.runner = runner

    def stop(self, service: str) -> None:
        """`docker compose stop <service>`."""
        self._compose("stop", service)

    def start(self, service: str) -> None:
        """`docker compose start <service>` (le healthcheck est attendu par l'appelant, sur HTTP)."""
        self._compose("start", service)

    def _compose(self, *args: str) -> None:
        command: Sequence[str] = ["docker", "compose", *args]
        try:
            done = self.runner(command, cwd=self.project_dir, capture_output=True, text=True,
                               timeout=COMPOSE_TIMEOUT_S, check=False)  # fmt: skip
        except (OSError, subprocess.TimeoutExpired) as exc:
            msg = f"`{' '.join(command)}` impossible : {type(exc).__name__} ({exc})"
            raise ServiceControlError(msg) from exc
        if done.returncode != 0:
            msg = f"`{' '.join(command)}` : code {done.returncode} — {(done.stderr or done.stdout).strip()[:300]}"
            raise ServiceControlError(msg)
