"""Suivre un alias sans redéploiement : ce qu'utilisera l'API (phase 7) pour servir `atp-winner@production`.

`refresh()` est appelé périodiquement : il lit le numéro de version désigné par l'alias et ne recharge le modèle que
s'il a changé (promotion ou rollback). Si MLflow ne répond pas, le dernier modèle chargé reste servi (scénario S8) et
le résultat le signale.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mlflow.exceptions import MlflowException

from tennis_mlops.registry.model_registry import ModelRegistry
from tennis_mlops.registry.tagging import Alias, now_iso


@dataclass(frozen=True)
class ServedModel:
    """Le modèle en mémoire et la version qu'il représente."""

    name: str
    version: str
    run_id: str
    model: Any
    loaded_at: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RefreshOutcome:
    """Ce qu'a fait un `refresh()`."""

    changed: bool
    version: str | None
    previous_version: str | None = None
    error: str | None = None  # MLflow injoignable ou alias absent : le modèle déjà chargé reste servi

    @property
    def stale(self) -> bool:
        """Vrai si l'état du registre n'a pas pu être vérifié."""
        return self.error is not None


class AliasModelLoader:
    """Garde en mémoire la version désignée par un alias et la remplace quand l'alias bouge."""

    def __init__(
        self,
        registry: ModelRegistry,
        alias: Alias = Alias.PRODUCTION,
        load: Callable[[str], Any] | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.registry = registry
        self.alias = alias
        self._load = load or registry.load
        self._clock = clock
        self._current: ServedModel | None = None
        self._lock = threading.Lock()

    @property
    def current(self) -> ServedModel | None:
        """Le modèle servi en ce moment (None avant le premier chargement réussi)."""
        with self._lock:
            return self._current

    def refresh(self) -> RefreshOutcome:
        """Relit l'alias ; recharge seulement si la version a changé."""
        served = self.current
        served_version = served.version if served else None
        try:
            target = self.registry.by_alias(self.alias)
            if target is None:
                return RefreshOutcome(False, served_version, error=f"alias `{self.alias}` posé sur aucune version")
            version = str(target.version)
            if version == served_version:
                return RefreshOutcome(False, version)
            model = self._load(version)  # par numéro : l'alias peut bouger pendant le chargement
        except (MlflowException, OSError) as exc:
            return RefreshOutcome(
                False, served_version, error=f"lecture du registre impossible, modèle conservé : {exc}"
            )
        loaded = ServedModel(self.registry.name, version, str(target.run_id), model, self._clock(), dict(target.tags))
        with self._lock:
            self._current = loaded
        return RefreshOutcome(True, version, previous_version=served_version)
