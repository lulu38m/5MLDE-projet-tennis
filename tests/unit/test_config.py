from pathlib import Path

import pytest
from pydantic import ValidationError

from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig, Settings, SourceKind


def write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


VALID_YAML = """
data:
  raw_dir: data/raw
  lock_file: configs/data.lock.json
  first_match_year: 2010
  last_match_year: 2024
  sources:
    - kind: huggingface
      repo: org/dataset
    - kind: github
      repo: org/mirror
      revision: v1
simulation:
  bootstrap_start: 2010-01-01
  replay_start: 2015-01-01
  replay_end: 2024-12-31
"""


def test_default_config_file_is_valid() -> None:
    config = ProjectConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert config.data.sources[0].kind is SourceKind.HUGGINGFACE
    assert config.data.sources[0].repo == "Aneeshers/tennis-sackmann-archive"
    assert config.simulation.replay_start.year == 2016


def test_source_defaults_and_name(tmp_path: Path) -> None:
    config = ProjectConfig.from_yaml(write_yaml(tmp_path, VALID_YAML))
    hf, gh = config.data.sources
    assert (hf.revision, hf.subdir) == ("main", "atp")
    assert gh.revision == "v1"
    assert hf.name == "huggingface:org/dataset"


def test_demo_section_is_optional(tmp_path: Path) -> None:
    config = ProjectConfig.from_yaml(write_yaml(tmp_path, VALID_YAML))
    assert config.demo.sleep_s == 0.0


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="extra"):
        ProjectConfig.from_yaml(write_yaml(tmp_path, VALID_YAML + "\ntypo_key: 1\n"))


def test_simulation_dates_must_be_chronological(tmp_path: Path) -> None:
    broken = VALID_YAML.replace("replay_start: 2015-01-01", "replay_start: 2030-01-01")
    with pytest.raises(ValidationError, match="bootstrap_start < replay_start < replay_end"):
        ProjectConfig.from_yaml(write_yaml(tmp_path, broken))


def test_simulation_must_fit_in_downloaded_years(tmp_path: Path) -> None:
    broken = VALID_YAML.replace("last_match_year: 2024", "last_match_year: 2020")
    with pytest.raises(ValidationError, match="fenêtre de simulation"):
        ProjectConfig.from_yaml(write_yaml(tmp_path, broken))


def test_year_range_must_be_ordered(tmp_path: Path) -> None:
    broken = VALID_YAML.replace("first_match_year: 2010", "first_match_year: 2030")
    with pytest.raises(ValidationError, match="first_match_year <= last_match_year"):
        ProjectConfig.from_yaml(write_yaml(tmp_path, broken))


def test_at_least_one_source_is_required(tmp_path: Path) -> None:
    head, _, tail = VALID_YAML.partition("  sources:")
    broken = head + "  sources: []\n" + tail[tail.index("simulation:") :]
    with pytest.raises(ValidationError, match="sources"):
        ProjectConfig.from_yaml(write_yaml(tmp_path, broken))


def test_invalid_repo_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="repo"):
        ProjectConfig.from_yaml(write_yaml(tmp_path, VALID_YAML.replace("org/mirror", "not a repo")))


def test_negative_demo_sleep_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.from_yaml(write_yaml(tmp_path, VALID_YAML + "demo:\n  sleep_s: -1\n"))


def test_yaml_root_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="objet YAML"):
        ProjectConfig.from_yaml(write_yaml(tmp_path, "- a\n- b\n"))


def test_relative_paths_are_resolved_from_root(tmp_path: Path) -> None:
    config = ProjectConfig.from_yaml(write_yaml(tmp_path, VALID_YAML))
    assert config.data.resolved_raw_dir(root=tmp_path) == tmp_path / "data" / "raw"
    assert config.data.resolved_lock_file(root=tmp_path) == tmp_path / "configs" / "data.lock.json"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    config = ProjectConfig.from_yaml(write_yaml(tmp_path, VALID_YAML.replace("data/raw", str(tmp_path / "raw"))))
    assert config.data.resolved_raw_dir(root=Path("/elsewhere")) == tmp_path / "raw"


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENNIS_MOCK_SOURCE_URL", "http://mock:9000")
    monkeypatch.setenv("TENNIS_TEAMS_WEBHOOK_URL", "https://example.invalid/hook")
    settings = Settings(_env_file=None)
    assert str(settings.mock_source_url) == "http://mock:9000/"
    assert settings.teams_webhook_url is not None
    assert "example.invalid" not in repr(settings)  # le secret n'apparaît jamais dans les logs


def test_shared_env_file_with_other_services_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MOCK_PORT=8001\nTENNIS_API_URL=http://api:9000\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert str(settings.api_url) == "http://api:9000/"
