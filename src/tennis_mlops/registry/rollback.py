"""Rollback : remettre en production la version que la dernière promotion a remplacée (alias `previous`).

Ne dépend que de MLflow (ni Prefect ni l'entraînement) : c'est la commande des incidents, elle doit marcher quand le
reste est en panne. L'API suit l'alias `production` : elle sert l'ancienne version sans redéploiement.
"""

from __future__ import annotations

from collections.abc import Callable

from tennis_mlops.registry.events import EventKind, RegistryEvent
from tennis_mlops.registry.model_registry import ModelRegistry
from tennis_mlops.registry.tagging import Alias, DecidedBy, Lifecycle, now_iso


class RollbackError(RuntimeError):
    """Rollback impossible : rien en production, pas de version `previous`, raison manquante."""


class Rollback:
    """`production` ← `previous`, avec la raison et l'heure sur les deux versions."""

    def __init__(self, registry: ModelRegistry, clock: Callable[[], str] = now_iso) -> None:
        self.registry = registry
        self.clock = clock

    def run(self, reason: str) -> RegistryEvent:
        """Effectue le rollback ; `previous` recule d'un cran si l'historique le permet, sinon il est retiré."""
        registry = self.registry
        if not reason.strip():
            msg = 'un rollback demande une raison (REASON="…")'
            raise RollbackError(msg)
        current = registry.by_alias(Alias.PRODUCTION)
        if current is None:
            msg = f"aucune version de {registry.name} en production : rien à annuler"
            raise RollbackError(msg)
        target = registry.by_alias(Alias.PREVIOUS)
        if target is None or str(target.version) == str(current.version):
            msg = f"aucune version `previous` pour {registry.name} : rien à restaurer (v{current.version} reste servie)"
            raise RollbackError(msg)
        at, bad, good = self.clock(), str(current.version), str(target.version)

        registry.set_alias(Alias.PRODUCTION, good)  # d'abord ce que l'API lit
        registry.tag(bad, {"lifecycle": Lifecycle.ROLLED_BACK, "rollback_reason": reason.strip(),
                           "rolled_back_at": at, "rolled_back_to": good})  # fmt: skip
        registry.tag(good, {"lifecycle": Lifecycle.PRODUCTION, "restored_at": at, "restored_from": bad})
        self._step_back_previous(target.tags.get("replaced_version", ""), bad)
        return RegistryEvent(EventKind.ROLLBACK, registry.name, good, bad, good, DecidedBy.MANUAL, reason.strip(), at)

    def _step_back_previous(self, earlier: str, rolled_back: str) -> None:
        """`previous` ← la version que la version restaurée avait remplacée : un 2e rollback reste possible."""
        registry = self.registry
        if earlier and earlier != rolled_back:
            candidate = registry.get(earlier)
            if candidate.tags.get("lifecycle") != Lifecycle.ROLLED_BACK:
                registry.set_alias(Alias.PREVIOUS, earlier)
                return
        registry.drop_alias(Alias.PREVIOUS)
