"""Le modèle servi : la version désignée par l'alias `production`, relue périodiquement et à la demande.

S'appuie sur `AliasModelLoader` (phase 5) : on ne recharge que si l'alias a bougé (promotion, rollback), et si MLflow
ne répond plus, la version déjà chargée reste servie et l'état passe à « périmé » (`/ready` dégradé, scénario S8).

Une version entraînée sur d'autres features (`features_version` différente de celle du code) n'est pas servie :
ses colonnes ne correspondraient pas à ce que calcule le `FeatureBuilder`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from tennis_mlops.features import FEATURES_VERSION
from tennis_mlops.registry.loader import AliasModelLoader, RefreshOutcome, ServedModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelStatus:
    """Où en est le modèle servi."""

    served: ServedModel | None
    last_check: datetime | None
    last_success: datetime | None
    error: str | None
    load_seconds: float | None
    incompatible: str | None = None  # raison pour laquelle la version de l'alias n'est pas servie

    @property
    def stale(self) -> bool:
        """Le registre n'a pas pu être relu au dernier essai : on sert peut-être une version qui n'est plus la bonne."""
        return self.error is not None


@dataclass
class ModelHolder:
    """Relit l'alias et garde le modèle servi ; ne lève jamais."""

    loader: AliasModelLoader
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _last_check: datetime | None = field(default=None, init=False)
    _last_success: datetime | None = field(default=None, init=False)
    _error: str | None = field(default=None, init=False)
    _load_seconds: float | None = field(default=None, init=False)
    _incompatible: str | None = field(default=None, init=False)
    reloads: int = field(default=0, init=False)

    @property
    def served(self) -> ServedModel | None:
        """Le modèle à utiliser (None : aucune version servable)."""
        served = self.loader.current
        if served is None or self._incompatible is not None:
            return None
        return served

    def status(self) -> ModelStatus:
        """Instantané de l'état, pour `/ready`, `/model` et les métriques."""
        with self._lock:
            return ModelStatus(self.served, self._last_check, self._last_success, self._error, self._load_seconds,
                               self._incompatible)  # fmt: skip

    def refresh(self) -> RefreshOutcome:
        """Relit l'alias ; recharge si la version a changé."""
        started = time.perf_counter()
        try:
            outcome = self.loader.refresh()
        except Exception as exc:
            served = self.loader.current
            outcome = RefreshOutcome(False, served.version if served else None, error=f"{type(exc).__name__} : {exc}")
        now = datetime.now(UTC)
        with self._lock:
            self._last_check = now
            self._error = outcome.error
            if outcome.error is None:
                self._last_success = now
            if outcome.changed:
                self.reloads += 1
                self._load_seconds = time.perf_counter() - started
                self._incompatible = _incompatibility(self.loader.current)
        if outcome.changed:
            logger.info("modèle servi : v%s (avant : %s)", outcome.version, outcome.previous_version or "aucun")
        if self._incompatible:
            logger.error("version non servie : %s", self._incompatible)
        if outcome.error:
            logger.warning("registre non relu : %s", outcome.error)
        return outcome


def _incompatibility(served: ServedModel | None) -> str | None:
    if served is None:
        return None
    trained_on = served.tags.get("features_version")
    if trained_on is not None and trained_on != FEATURES_VERSION:
        return (f"{served.name} v{served.version} entraînée avec les features v{trained_on}, "
                f"le code calcule les features v{FEATURES_VERSION} : réentraîner (make train)")  # fmt: skip
    return None
