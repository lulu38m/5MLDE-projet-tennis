"""Appliquer une décision : déplacer les alias et mettre à jour le cycle de vie des versions concernées.

Ordre des écritures pour une promotion (MLflow n'a pas de transaction sur les alias) :
1. `previous` ← l'ancienne production ; 2. `production` ← la nouvelle version ; 3. tags de cycle de vie.
À chaque instant, `production` désigne une version valide : une coupure au milieu ne laisse jamais l'API sans modèle.
"""

from __future__ import annotations

from collections.abc import Callable

from tennis_mlops.registry.events import EventKind, RegistryEvent
from tennis_mlops.registry.gate import GateDecision
from tennis_mlops.registry.model_registry import ModelRegistry
from tennis_mlops.registry.tagging import Alias, DecidedBy, Lifecycle, now_iso


class PromotionError(RuntimeError):
    """Promotion impossible (version déjà en production, raison manquante…)."""


class Promoter:
    """Promotions (automatiques ou manuelles) et rejets."""

    def __init__(self, registry: ModelRegistry, clock: Callable[[], str] = now_iso) -> None:
        self.registry = registry
        self.clock = clock

    def apply(self, version: str, decision: GateDecision, campaign_run_id: str | None = None) -> RegistryEvent:
        """Verdict de la campagne : promotion si toutes les règles passent, rejet sinon."""
        if decision.passed:
            return self._promote(version, EventKind.PROMOTION, DecidedBy.PIPELINE, decision.reason, decision,
                                 campaign_run_id)  # fmt: skip
        return self._reject(version, decision, campaign_run_id)

    def promote_manually(self, version: str, reason: str) -> RegistryEvent:
        """`make promote` : une personne met une version en production, raison obligatoire (la règle est contournée)."""
        if not reason.strip():
            msg = 'une promotion manuelle demande une raison (REASON="…")'
            raise PromotionError(msg)
        self.registry.get(version)
        return self._promote(version, EventKind.MANUAL_PROMOTION, DecidedBy.MANUAL, reason.strip(), None, None)

    # ------------------------------------------------------------------ cœur

    def _promote(
        self,
        version: str,
        kind: EventKind,
        decided_by: DecidedBy,
        reason: str,
        decision: GateDecision | None,
        campaign_run_id: str | None,
    ) -> RegistryEvent:
        registry, at = self.registry, self.clock()
        current = registry.by_alias(Alias.PRODUCTION)
        before = str(current.version) if current else None
        if before == version:
            msg = f"{registry.name} v{version} est déjà en production"
            raise PromotionError(msg)
        if before:
            registry.set_alias(Alias.PREVIOUS, before)
        registry.set_alias(Alias.PRODUCTION, version)
        if before:
            registry.tag(before, {"lifecycle": Lifecycle.ARCHIVED, "archived_at": at, "replaced_by": version})
        tags: dict[str, object] = {
            "lifecycle": Lifecycle.PRODUCTION,
            "promoted_at": at,
            "replaced_version": before or "",
            "decided_by": decided_by,
            "decided_at": at,
        }
        if decision is not None:
            tags.update(decision.tags())
        if decided_by is DecidedBy.MANUAL:
            tags["promotion_reason"] = reason
        registry.tag(version, tags)
        staging = registry.by_alias(Alias.STAGING)
        if staging is not None and int(staging.version) <= int(version):
            registry.drop_alias(Alias.STAGING)  # plus rien à observer : la production est au moins aussi récente
        return RegistryEvent(kind, registry.name, version, before, version, decided_by, reason, at, decision,
                             campaign_run_id)  # fmt: skip

    def _reject(self, version: str, decision: GateDecision, campaign_run_id: str | None) -> RegistryEvent:
        registry, at = self.registry, self.clock()
        current = registry.by_alias(Alias.PRODUCTION)
        before = str(current.version) if current else None
        registry.tag(version, {"lifecycle": Lifecycle.REJECTED, "decided_by": DecidedBy.PIPELINE, "decided_at": at,
                               **decision.tags()})  # fmt: skip
        registry.set_alias(Alias.STAGING, version)
        return RegistryEvent(EventKind.REJECTION, registry.name, version, before, before, DecidedBy.PIPELINE,
                             decision.reason, at, decision, campaign_run_id)  # fmt: skip
