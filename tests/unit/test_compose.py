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


def test_the_worker_writes_state_as_the_host_user() -> None:
    """Sous Linux et WSL2, ./state garde les droits du disque : le worker doit écrire sous la même identité que
    `make ingest` lancé depuis l'hôte (sinon PermissionError dans state/gx dès le premier pas de `make demo`)."""
    worker = COMPOSE["services"]["worker"]
    assert worker["user"] == "${HOST_UID:-10001}:${HOST_GID:-10001}"
    assert "./state:/app/state" in worker["volumes"]
    assert worker["environment"]["PREFECT_HOME"].startswith("/tmp/")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "export HOST_UID HOST_GID" in makefile
    assert "@mkdir -p state" in makefile  # avant `docker compose up`, sinon Docker le crée au nom de root
