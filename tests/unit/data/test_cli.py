import shutil
from pathlib import Path

import pytest

from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig
from tennis_mlops.data.__main__ import EXIT_INTEGRITY, EXIT_OK, EXIT_UNAVAILABLE, default_sources, main
from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.lockfile import DataLock
from tennis_mlops.data.sources import GitHubSource, HuggingFaceSource

from .fakes import InMemorySource
from .test_profile import write_raw_files


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Projet minimal : la config par défaut réduite à 2019-2020, dans un dossier temporaire."""
    (tmp_path / "configs").mkdir()
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace("first_match_year: 2010", "first_match_year: 2019")
    text = text.replace("last_match_year: 2026", "last_match_year: 2020")
    text = text.replace("bootstrap_start: 2010-01-01", "bootstrap_start: 2019-01-01")
    text = text.replace("replay_start: 2016-01-01", "replay_start: 2019-06-01")
    text = text.replace("replay_end: 2026-05-31", "replay_end: 2020-12-31")
    (tmp_path / "configs" / "project.yaml").write_text(text, encoding="utf-8")
    return tmp_path


def contents_from_fixture(tmp_path: Path) -> dict[str, bytes]:
    fixture = tmp_path / "fixture"
    catalog = write_raw_files(fixture)
    (fixture / "matches_data_dictionary.txt").write_text("dictionnaire")
    (fixture / "UPSTREAM_README.md").write_text("readme")
    contents = {name: (fixture / name).read_bytes() for name in catalog.names()}
    shutil.rmtree(fixture)
    return contents


def run(project: Path, command: str, *sources: InMemorySource) -> int:
    return main([command, "--config", str(project / "configs" / "project.yaml")], source_factory=lambda _: [*sources])


def test_lock_download_verify_profile_cycle(project: Path) -> None:
    contents = contents_from_fixture(project)
    hf = InMemorySource("huggingface:Aneeshers/tennis-sackmann-archive", contents, available=False)
    gh = InMemorySource("github:Aneeshers/tennis-sackmann-archive", contents, revision="c" * 40)

    assert run(project, "lock", hf, gh) == EXIT_OK
    lock = DataLock.load(project / "configs" / "data.lock.json")
    assert lock.revisions == {"github:Aneeshers/tennis-sackmann-archive": "c" * 40}

    assert run(project, "download", hf, gh) == EXIT_OK
    assert run(project, "verify", hf, gh) == EXIT_OK
    assert run(project, "profile", hf, gh) == EXIT_OK
    assert (project / "data" / "raw" / "profile.json").is_file()

    (project / "data" / "raw" / "atp_players.csv").write_text("corrompu")
    assert run(project, "verify", hf, gh) == EXIT_INTEGRITY


def test_download_without_lock_still_works(project: Path) -> None:
    contents = contents_from_fixture(project)
    assert run(project, "download", InMemorySource("hf", contents)) == EXIT_OK


def test_no_source_available_exit_code(project: Path) -> None:
    offline = InMemorySource("hf", {}, available=False)
    assert run(project, "download", offline) == EXIT_UNAVAILABLE


def test_default_sources_follow_config_order() -> None:
    sources = default_sources(ProjectConfig.from_yaml())
    assert [type(s) for s in sources] == [HuggingFaceSource, GitHubSource]
    assert DatasetCatalog.from_config(ProjectConfig.from_yaml().data).names()
