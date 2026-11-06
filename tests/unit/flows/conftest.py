"""Les flows tournent contre un serveur Prefect de test (base SQLite temporaire), démarré une fois par session."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

from tennis_mlops.flows import ingest
from tennis_mlops.ingestion.archive import LotArchive
from tennis_mlops.ingestion.client import MockSourceClient
from tennis_mlops.ingestion.store import IngestionStore
from tennis_mlops.validation.validator import LotValidator
from tests.support.mock_world import MockEnv, write_project_config


@pytest.fixture(scope="session", autouse=True)
def prefect_backend() -> Iterator[None]:
    with prefect_test_harness(server_startup_timeout=120):
        yield


@dataclass
class FlowWorld:
    """Un projet de test complet : simulateur, base SQLite, état local, configuration."""

    env: MockEnv
    config_path: Path
    state_dir: Path
    store: IngestionStore


@pytest.fixture
def world(env: MockEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FlowWorld:
    state_dir = tmp_path / "state"
    config_path = write_project_config(tmp_path, state_dir=state_dir)
    store = IngestionStore.from_url(f"sqlite:///{tmp_path / 'tennis.sqlite'}")

    def dependencies(*_: object) -> ingest.IngestionDependencies:
        return ingest.IngestionDependencies(
            client=MockSourceClient("", 5, http=env.client),
            store=store,
            validator=LotValidator(state_dir / "gx"),
            archive=LotArchive(state_dir),
        )

    monkeypatch.setattr(ingest, "dependency_factory", dependencies)
    return FlowWorld(env=env, config_path=config_path, state_dir=state_dir, store=store)
