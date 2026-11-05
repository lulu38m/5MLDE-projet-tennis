"""Assemblage du service à partir de la configuration du projet (utilisé par uvicorn et Docker)."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from mock_source.app import create_app
from mock_source.chaos import ChaosController
from mock_source.clock import SimulatedClock
from mock_source.repository import ReplayRepository
from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig
from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.errors import DataError
from tennis_mlops.data.lockfile import DataLock
from tennis_mlops.log import configure_logging

logger = logging.getLogger("mock_source")


class MockSettings(BaseSettings):
    """Paramètres d'exécution du simulateur (variables `MOCK_*`)."""

    model_config = SettingsConfigDict(env_prefix="MOCK_", env_file=".env", extra="ignore", frozen=True)

    config_path: Path = DEFAULT_CONFIG_PATH
    start_date: date | None = None
    speed: float = 0.0
    host: str = "0.0.0.0"  # écoute dans le conteneur ; compose ne publie que sur 127.0.0.1
    port: int = 8001


def build_app(settings: MockSettings | None = None) -> FastAPI:
    """Charge et vérifie les données verrouillées, règle l'horloge et construit l'API.

    Raises:
        SystemExit: données absentes, illisibles ou altérées ; le message d'erreur dit quoi faire.
    """
    configure_logging()
    settings = settings or MockSettings()
    config = ProjectConfig.from_yaml(settings.config_path)
    root = settings.config_path.resolve().parents[1]
    try:
        lock = DataLock.load(config.data.resolved_lock_file(root))
        repository = ReplayRepository.from_raw(
            config.data.resolved_raw_dir(root),
            DatasetCatalog.from_config(config.data),
            lock,
            config.mock_source,
        )
    except (DataError, FileNotFoundError) as exc:
        logger.error("Démarrage impossible : %s", exc)  # message actionnable, sans trace Python
        raise SystemExit(1) from exc
    simulation = config.simulation
    clock = SimulatedClock(
        start=settings.start_date or simulation.replay_start,
        lower=simulation.bootstrap_start,
        upper=simulation.replay_end + timedelta(days=1),
        speed=settings.speed,
    )
    return create_app(repository, clock, ChaosController(), config.mock_source.fixture_horizon_days)
