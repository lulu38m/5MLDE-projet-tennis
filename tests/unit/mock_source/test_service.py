import logging
import shutil
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock_source.repository import ReplayRepository, check_readable
from mock_source.service import MockSettings, build_app
from tennis_mlops.config import DEFAULT_CONFIG_PATH
from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.errors import DataAccessError
from tennis_mlops.data.lockfile import DataLock
from tests.support.mock_world import CONFIG


def make_project(tmp_path: Path, dataset: tuple[Path, DatasetCatalog, DataLock]) -> Path:
    """Projet minimal pointant vers le mini-dataset : config 2019-2020 + verrou."""
    raw, _, lock = dataset
    project_raw = tmp_path / "data" / "raw"
    shutil.copytree(raw, project_raw)
    (tmp_path / "configs").mkdir()
    lock.save(tmp_path / "configs" / "data.lock.json")
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    for old, new in [
        ("first_match_year: 2010", "first_match_year: 2019"),
        ("last_match_year: 2026", "last_match_year: 2020"),
        ("bootstrap_start: 2010-01-01", "bootstrap_start: 2019-01-01"),
        ("replay_start: 2015-01-01", "replay_start: 2019-01-05"),
        ("replay_end: 2026-05-31", "replay_end: 2020-12-31"),
    ]:
        text = text.replace(old, new)
    config = tmp_path / "configs" / "project.yaml"
    config.write_text(text, encoding="utf-8")
    return config


def test_build_app_from_project_configuration(tmp_path: Path, dataset: tuple[Path, DatasetCatalog, DataLock]) -> None:
    config = make_project(tmp_path, dataset)
    app = build_app(MockSettings(config_path=config, start_date=date(2019, 1, 8), _env_file=None))
    with TestClient(app) as client:
        assert client.get("/health").json()["simulated_now"] == "2019-01-08"
        assert client.get("/fixtures").json()["count"] == 2


def test_default_start_is_replay_start(tmp_path: Path, dataset: tuple[Path, DatasetCatalog, DataLock]) -> None:
    config = make_project(tmp_path, dataset)
    with TestClient(build_app(MockSettings(config_path=config, _env_file=None))) as client:
        assert client.get("/admin/clock").json()["now"] == "2019-01-05"


def test_altered_data_stops_startup_with_clear_message(
    tmp_path: Path, dataset: tuple[Path, DatasetCatalog, DataLock], caplog: pytest.LogCaptureFixture
) -> None:
    config = make_project(tmp_path, dataset)
    (tmp_path / "data" / "raw" / "atp_players.csv").write_text("altéré")
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as excinfo:
        build_app(MockSettings(config_path=config, _env_file=None))
    assert excinfo.value.code == 1
    assert "Démarrage impossible" in caplog.text and "atp_players.csv" in caplog.text


def test_missing_lock_stops_startup(tmp_path: Path, dataset: tuple[Path, DatasetCatalog, DataLock]) -> None:
    config = make_project(tmp_path, dataset)
    (tmp_path / "configs" / "data.lock.json").unlink()
    with pytest.raises(SystemExit):
        build_app(MockSettings(config_path=config, _env_file=None))


def test_unreadable_files_give_an_actionable_error(
    dataset: tuple[Path, DatasetCatalog, DataLock], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, catalog, lock = dataset
    monkeypatch.setattr("mock_source.repository.os.access", lambda path, mode: not str(path).endswith(".csv"))
    with pytest.raises(DataAccessError, match=r"chmod a\+r data/raw/\*"):
        check_readable(raw, catalog.names())
    with pytest.raises(DataAccessError):
        ReplayRepository.from_raw(raw, catalog, lock, CONFIG)
