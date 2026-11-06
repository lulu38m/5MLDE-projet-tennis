"""MLflow local pour les tests : base SQLite et artefacts dans un dossier temporaire (pas de `mlruns` dans le dépôt)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    """MLflow range les artefacts d'un backend SQLite sous `./mlruns` : on se place dans le dossier de test."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def sqlite_uri(folder: Path) -> str:
    """URI de suivi SQLite dans `folder`."""
    return f"sqlite:///{folder / 'mlflow.db'}"
