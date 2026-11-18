"""Garde-fous sur compose.yml : ce qu'un incident nous a appris."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = yaml.safe_load((ROOT / "compose.yml").read_text(encoding="utf-8"))


def _locked(name: str) -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return str(next(p["version"] for p in lock["package"] if p["name"] == name))


def test_prefect_server_runs_the_locked_dependencies() -> None:
    """L'image prefecthq/prefect embarque SQLAlchemy 2.1 : le planificateur de Prefect 3.8.7 n'y crée aucun run sur
    PostgreSQL. Le serveur tourne donc dans notre image, avec les versions de uv.lock."""
    prefect = COMPOSE["services"]["prefect"]
    assert prefect["image"] == COMPOSE["services"]["api"]["image"] == "tennis-mlops:local"
    assert prefect["command"][:3] == ["prefect", "server", "start"]
    # L'utilisateur de l'image ne peut pas écrire dans le paquet Prefect : l'interface est copiée dans /tmp.
    assert prefect["environment"]["PREFECT_UI_STATIC_DIRECTORY"].startswith("/tmp/")
    assert _locked("prefect") == "3.8.7"
    assert _locked("sqlalchemy").startswith("2.0.")
    assert _locked("asyncpg")  # pilote du backend PostgreSQL de Prefect
