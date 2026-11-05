"""Configuration du projet, séparée du code.

Deux sources, deux responsabilités :

- `Settings` : ce qui dépend de l'environnement d'exécution (URLs des services, secrets). Lu depuis les
  variables d'environnement ou un fichier `.env` jamais commité.
- `ProjectConfig` : ce qui décrit l'expérience (données, simulation, démo). Lu depuis `configs/*.yaml`,
  versionné dans Git et tracé dans MLflow.

Les deux modèles refusent les clés inconnues : une faute de frappe dans la configuration fait échouer le
lancement au lieu d'être ignorée silencieusement.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.yaml"


class Settings(BaseSettings):
    """Paramètres d'environnement : où sont les services et avec quels identifiants."""

    model_config = SettingsConfigDict(env_prefix="TENNIS_", env_file=".env", extra="forbid", frozen=True)

    mlflow_tracking_uri: HttpUrl = HttpUrl("http://localhost:7281")
    prefect_api_url: HttpUrl = HttpUrl("http://localhost:4200/api")
    mock_source_url: HttpUrl = HttpUrl("http://localhost:8001")
    api_url: HttpUrl = HttpUrl("http://localhost:8000")
    postgres_dsn: SecretStr = SecretStr("postgresql://app_writer:change-me@localhost:5432/app")
    teams_webhook_url: SecretStr | None = None


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_StrictModel):
    """Provenance des données brutes."""

    hf_repo_id: str
    hf_revision: str = Field(min_length=1, description="Révision Hugging Face figée (sha ou tag)")
    allow_patterns: tuple[str, ...] = Field(min_length=1)
    raw_dir: Path

    def resolved_raw_dir(self, root: Path = PROJECT_ROOT) -> Path:
        """Chemin absolu du dossier des données brutes."""
        return self.raw_dir if self.raw_dir.is_absolute() else root / self.raw_dir


class SimulationConfig(_StrictModel):
    """Fenêtre de rejeu du dataset historique."""

    bootstrap_start: date
    replay_start: date
    replay_end: date

    @model_validator(mode="after")
    def _check_chronology(self) -> Self:
        if not self.bootstrap_start < self.replay_start < self.replay_end:
            msg = "attendu : bootstrap_start < replay_start < replay_end"
            raise ValueError(msg)
        return self


class DemoConfig(_StrictModel):
    """Réglages qui rendent la démo lisible, sans effet sur les résultats."""

    sleep_s: float = Field(default=0.0, ge=0.0, le=60.0)


class ProjectConfig(_StrictModel):
    """Configuration complète d'une exécution."""

    data: DataConfig
    simulation: SimulationConfig
    demo: DemoConfig = DemoConfig()

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> Self:
        """Charge et valide un fichier YAML de configuration."""
        with path.open(encoding="utf-8") as handle:
            raw: object = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            msg = f"{path} doit contenir un objet YAML à la racine"
            raise TypeError(msg)
        return cls.model_validate(raw)
