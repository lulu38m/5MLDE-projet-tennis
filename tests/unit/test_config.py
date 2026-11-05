from pathlib import Path

import pytest
from pydantic import ValidationError

from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig, Settings


def write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


VALID_YAML = """
data:
  hf_repo_id: org/dataset
  hf_revision: abc123
  allow_patterns: ["atp/*.csv"]
  raw_dir: data/raw
simulation:
  bootstrap_start: 2010-01-01
  replay_start: 2015-01-01
  replay_end: 2024-12-31
"""


def test_default_config_file_is_valid() -> None:
    config = ProjectConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert config.data.hf_repo_id == "Aneeshers/tennis-sackmann-archive"
    assert config.simulation.replay_start.year == 2015


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


def test_negative_demo_sleep_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.from_yaml(write_yaml(tmp_path, VALID_YAML + "demo:\n  sleep_s: -1\n"))


def test_yaml_root_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="objet YAML"):
        ProjectConfig.from_yaml(write_yaml(tmp_path, "- a\n- b\n"))


def test_relative_raw_dir_is_resolved_from_root(tmp_path: Path) -> None:
    config = ProjectConfig.from_yaml(write_yaml(tmp_path, VALID_YAML))
    assert config.data.resolved_raw_dir(root=tmp_path) == tmp_path / "data" / "raw"


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENNIS_MOCK_SOURCE_URL", "http://mock:9000")
    monkeypatch.setenv("TENNIS_TEAMS_WEBHOOK_URL", "https://example.invalid/hook")
    settings = Settings(_env_file=None)
    assert str(settings.mock_source_url) == "http://mock:9000/"
    assert settings.teams_webhook_url is not None
    assert "example.invalid" not in repr(settings)  # le secret n'apparaît jamais dans les logs
