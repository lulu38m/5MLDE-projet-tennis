"""Fixtures partagées : le mini-dataset synthétique et un simulateur `mock_source` prêt à l'emploi."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock_source.app import create_app
from mock_source.chaos import ChaosController
from mock_source.clock import SimulatedClock
from mock_source.repository import ReplayRepository
from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.lockfile import DataLock
from tests.support.mock_world import CONFIG, MockEnv, write_dataset


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, DatasetCatalog, DataLock]:
    raw = tmp_path_factory.mktemp("mock") / "raw"
    catalog, lock = write_dataset(raw)
    return raw, catalog, lock


@pytest.fixture(scope="session")
def repository(dataset: tuple[Path, DatasetCatalog, DataLock]) -> ReplayRepository:
    raw, catalog, lock = dataset
    return ReplayRepository.from_raw(raw, catalog, lock, CONFIG)


@pytest.fixture
def env(repository: ReplayRepository, dataset: tuple[Path, DatasetCatalog, DataLock]) -> Iterator[MockEnv]:
    raw, catalog, lock = dataset
    clock = SimulatedClock(start=date(2019, 1, 1), lower=date(2018, 12, 1), upper=date(2020, 12, 31))
    chaos = ChaosController()
    with TestClient(create_app(repository, clock, chaos, CONFIG.fixture_horizon_days)) as client:
        yield MockEnv(repository, clock, chaos, client, raw, catalog, lock)
