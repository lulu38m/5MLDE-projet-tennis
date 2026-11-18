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

import platform
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.yaml"
FIRST_ATP_YEAR = 1968


class Settings(BaseSettings):
    """Paramètres d'environnement : où sont les services et avec quels identifiants."""

    model_config = SettingsConfigDict(env_prefix="TENNIS_", env_file=".env", extra="ignore", frozen=True)

    # Ports d'hôte décalés (5433, 4201) pour coexister avec la stack des TP (5432, 4200) si elle tourne encore.
    mlflow_tracking_uri: HttpUrl = HttpUrl("http://127.0.0.1:7282")
    prefect_api_url: HttpUrl = HttpUrl("http://127.0.0.1:4201/api")
    mock_source_url: HttpUrl = HttpUrl("http://127.0.0.1:8001")
    api_url: HttpUrl = HttpUrl("http://127.0.0.1:8000")
    # Identifiants de démonstration d'une base locale liée à 127.0.0.1 ; surchargés par .env.
    postgres_dsn: SecretStr = SecretStr("postgresql://tennis:tennis-local@127.0.0.1:5433/tennis")
    teams_webhook_url: SecretStr | None = None
    # Phase 6 : service d'alertes, et adresses des interfaces telles que les voit le navigateur (les liens des cartes
    # Teams), qui diffèrent des adresses internes quand un service tourne dans Docker.
    alert_sink_url: HttpUrl = HttpUrl("http://127.0.0.1:8090")
    prefect_ui_url: HttpUrl = HttpUrl("http://127.0.0.1:4201")
    mlflow_ui_url: HttpUrl = HttpUrl("http://127.0.0.1:7282")
    alert_host: str = Field(default="", description="Poste affiché dans les cartes (défaut : nom de la machine)")
    alert_owner: str = Field(default="", description="Prénom affiché dans les cartes, dans un canal partagé")
    # Phase 7 : l'API écrit avec son propre compte (app_writer), qui ne peut que lire les tables d'ingestion.
    api_postgres_dsn: SecretStr = SecretStr("postgresql://app_writer:app-writer-local@127.0.0.1:5433/tennis")
    admin_token: SecretStr | None = Field(default=None, description="Si défini, exigé par POST /admin/reload")
    # Phase 8 : liens de l'interface de test, ouverts par le navigateur (adresses publiées sur l'hôte).
    api_ui_url: HttpUrl = HttpUrl("http://127.0.0.1:8000")
    grafana_ui_url: HttpUrl = HttpUrl("http://127.0.0.1:3001")
    # Phase 9 : Prometheus et le compte admin de Grafana (le même mot de passe est donné à Grafana par compose), pour
    # `make grafana-check` depuis la machine hôte.
    prometheus_url: HttpUrl = HttpUrl("http://127.0.0.1:9091")
    grafana_admin_password: SecretStr = SecretStr("admin-local")
    # Phase 11 : l'interface de test vue depuis la machine hôte (`make scenarios`, S9 ; `make urls`).
    ui_url: HttpUrl = HttpUrl("http://127.0.0.1:8501")

    def alert_origin(self) -> str:
        """Le poste qui envoie : `TENNIS_ALERT_HOST` (le Makefile y met le nom du Mac), sinon le nom de la machine."""
        return self.alert_host.strip() or platform.node()

    def sqlalchemy_url(self) -> str:
        """URL SQLAlchemy (pilote psycopg 3) à partir du DSN PostgreSQL ; les autres schémas passent tels quels."""
        return _sqlalchemy(self.postgres_dsn)

    def api_sqlalchemy_url(self) -> str:
        """Même chose pour le compte de l'API."""
        return _sqlalchemy(self.api_postgres_dsn)


def _sqlalchemy(dsn_secret: SecretStr) -> str:
    dsn = dsn_secret.get_secret_value()
    for prefix in ("postgresql://", "postgres://"):
        if dsn.startswith(prefix):
            return "postgresql+psycopg://" + dsn.removeprefix(prefix)
    return dsn


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


class IngestionConfig(_StrictModel):
    """Robustesse des appels à la source et emplacement des traces locales."""

    http_timeout_s: float = Field(default=10.0, gt=0, le=120)
    retries: int = Field(default=3, ge=0, le=10)
    retry_delays_s: tuple[float, ...] = Field(default=(1.0, 2.0, 4.0), description="Backoff entre deux essais")
    state_dir: Path = Path("state")
    drift_min_history: int = Field(default=20, ge=3, description="Tournois de même niveau requis pour la dérive")
    drift_quantiles: tuple[float, float] = (0.005, 0.995)

    @model_validator(mode="after")
    def _check_quantiles(self) -> Self:
        low, high = self.drift_quantiles
        if not 0 <= low < high <= 1:
            msg = "drift_quantiles : attendu 0 <= bas < haut <= 1"
            raise ValueError(msg)
        return self

    def resolved_state_dir(self, root: Path = PROJECT_ROOT) -> Path:
        """Dossier des traces locales (zone brute, quarantaine, rapports Great Expectations)."""
        return resolve_path(self.state_dir, root)


AlertKindName = Literal["echec", "quarantaine", "degrade", "rollback", "promotion"]


class OrchestrationConfig(_StrictModel):
    """Chaîne automatique : simulation planifiée, entraînement déclenché par événement, alertes Teams."""

    simulation_days_per_run: int = Field(default=7, ge=1, le=90, description="Jours simulés par exécution")
    simulation_interval_s: int = Field(default=120, ge=30, description="Une exécution toutes les N secondes réelles")
    training_trigger_matches: int = Field(
        default=200, ge=0, description="Nouveaux matchs avant réentraînement (0 = jamais)"
    )
    teams_events: tuple[AlertKindName, ...] = ("echec", "quarantaine", "degrade", "rollback")
    alert_cooldown_min: int = Field(
        default=30, ge=0, description="Même incident : pas de nouvelle carte avant N minutes (0 = toujours envoyer)"
    )


class ApiConfig(_StrictModel):
    """API de prédiction (phase 7)."""

    model_refresh_s: int = Field(default=30, ge=5, description="Relecture de l'alias production toutes les N s")
    features_refresh_s: int = Field(default=30, ge=5, description="Vérification des nouveaux matchs toutes les N s")
    max_batch: int = Field(default=200, ge=1, le=1000, description="Matchs au plus par POST /predict/batch")
    write_queue: int = Field(default=10_000, ge=100, description="Lignes en attente d'écriture avant d'en perdre")
    write_flush_s: float = Field(default=0.5, gt=0, le=10, description="Écriture en base par paquets, toutes les N s")
    drift_window: int = Field(default=500, ge=50, description="Dernières prédictions comparées à l'entraînement")
    drift_reference_days: int = Field(default=730, ge=90, description="Période d'entraînement de référence (jours)")


class DemoConfig(_StrictModel):
    """Réglages qui rendent la démo lisible, sans effet sur les résultats."""

    sleep_s: float = Field(default=0.0, ge=0.0, le=60.0)


class ProjectConfig(_StrictModel):
    """Configuration complète d'une exécution."""

    data: DataConfig
    simulation: SimulationConfig
    mock_source: MockSourceConfig = MockSourceConfig()
    ingestion: IngestionConfig = IngestionConfig()
    demo: DemoConfig = DemoConfig()
    orchestration: OrchestrationConfig = OrchestrationConfig()
    api: ApiConfig = ApiConfig()

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
