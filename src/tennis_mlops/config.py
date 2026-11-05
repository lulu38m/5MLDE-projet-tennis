"""Configuration du projet, séparée du code.

Deux sources, deux responsabilités :

- `Settings` : ce qui dépend de l'environnement d'exécution (URLs des services, secrets). Lu depuis les
  variables d'environnement ou un fichier `.env` jamais commité.
- `ProjectConfig` : ce qui décrit l'expérience (données, simulation, démo). Lu depuis `configs/*.yaml`,
  versionné dans Git et tracé dans MLflow.

`ProjectConfig` refuse les clés inconnues : une faute de frappe dans le YAML fait échouer le lancement au lieu
d'être ignorée silencieusement. `Settings` ignore en revanche les variables qui ne la concernent pas, car le même
`.env` sert à tous les services de la stack (`MOCK_*` pour le simulateur, `TENNIS_*` pour la chaîne MLOps…).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.yaml"
FIRST_ATP_YEAR = 1968


class Settings(BaseSettings):
    """Paramètres d'environnement : où sont les services et avec quels identifiants."""

    model_config = SettingsConfigDict(env_prefix="TENNIS_", env_file=".env", extra="ignore", frozen=True)

    mlflow_tracking_uri: HttpUrl = HttpUrl("http://localhost:7281")
    prefect_api_url: HttpUrl = HttpUrl("http://localhost:4200/api")
    mock_source_url: HttpUrl = HttpUrl("http://localhost:8001")
    api_url: HttpUrl = HttpUrl("http://localhost:8000")
    postgres_dsn: SecretStr = SecretStr("postgresql://app_writer:change-me@localhost:5432/app")
    teams_webhook_url: SecretStr | None = None


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def resolve_path(path: Path, root: Path = PROJECT_ROOT) -> Path:
    """Rend un chemin de configuration absolu, relativement à la racine du projet."""
    return path if path.is_absolute() else root / path


class SourceKind(StrEnum):
    """Hébergeurs du dataset supportés."""

    HUGGINGFACE = "huggingface"
    GITHUB = "github"


class SourceConfig(_StrictModel):
    """Un emplacement d'où télécharger les fichiers bruts. L'ordre dans la liste définit le fallback."""

    kind: SourceKind
    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$", description="propriétaire/nom")
    revision: str = Field(default="main", min_length=1, description="Branche, tag ou sha à résoudre")
    subdir: str = "atp"
    timeout_s: float = Field(default=60.0, gt=0)

    @property
    def name(self) -> str:
        """Identifiant stable de la source, utilisé comme clé dans le fichier de verrouillage."""
        return f"{self.kind.value}:{self.repo}"


class DataConfig(_StrictModel):
    """Provenance et périmètre des données brutes."""

    raw_dir: Path
    lock_file: Path
    first_match_year: int = Field(ge=FIRST_ATP_YEAR)
    last_match_year: int
    sources: tuple[SourceConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_years(self) -> Self:
        if self.first_match_year > self.last_match_year:
            msg = "attendu : first_match_year <= last_match_year"
            raise ValueError(msg)
        return self

    def resolved_raw_dir(self, root: Path = PROJECT_ROOT) -> Path:
        """Chemin absolu du dossier des données brutes."""
        return resolve_path(self.raw_dir, root)

    def resolved_lock_file(self, root: Path = PROJECT_ROOT) -> Path:
        """Chemin absolu du fichier de verrouillage des données."""
        return resolve_path(self.lock_file, root)


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


class PauseConfig(_StrictModel):
    """Période sans tournoi connue, expliquée par `mock_source` quand on l'interroge."""

    start: date
    end: date
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.start > self.end:
            msg = "pause : start doit précéder end"
            raise ValueError(msg)
        return self


class MockSourceConfig(_StrictModel):
    """Règles de visibilité du simulateur de source."""

    first_round_notice_days: int = Field(default=2, ge=0, le=14, description="Tableau publié N jours avant")
    tournament_notice_days: int = Field(default=14, ge=0, le=90, description="Tournoi visible N jours avant")
    fixture_horizon_days: int = Field(default=7, ge=1, le=30, description="Horizon par défaut de /fixtures")
    pauses: tuple[PauseConfig, ...] = ()


class DemoConfig(_StrictModel):
    """Réglages qui rendent la démo lisible, sans effet sur les résultats."""

    sleep_s: float = Field(default=0.0, ge=0.0, le=60.0)


class ProjectConfig(_StrictModel):
    """Configuration complète d'une exécution."""

    data: DataConfig
    simulation: SimulationConfig
    mock_source: MockSourceConfig = MockSourceConfig()
    demo: DemoConfig = DemoConfig()

    @model_validator(mode="after")
    def _check_simulation_within_data(self) -> Self:
        if not (
            self.data.first_match_year <= self.simulation.bootstrap_start.year
            and self.simulation.replay_end.year <= self.data.last_match_year
        ):
            msg = "la fenêtre de simulation doit être couverte par les années de matchs téléchargées"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> Self:
        """Charge et valide un fichier YAML de configuration."""
        with path.open(encoding="utf-8") as handle:
            raw: object = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            msg = f"{path} doit contenir un objet YAML à la racine"
            raise TypeError(msg)
        return cls.model_validate(raw)
