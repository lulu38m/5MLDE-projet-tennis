"""Promotions, rejets, rollbacks et journal, sur un vrai registre MLflow (SQLite)."""

from collections.abc import Callable, Iterator
from itertools import count
from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException

from tennis_mlops.registry.events import EventKind
from tennis_mlops.registry.model_registry import UnknownVersionError
from tennis_mlops.registry.promotion import Promoter, PromotionError
from tennis_mlops.registry.rollback import Rollback, RollbackError
from tests.support.registry import RegistryWorld, registry_world, verdict


@pytest.fixture
def world(tmp_path: Path) -> Iterator[RegistryWorld]:
    with registry_world(tmp_path) as w:
        yield w


def clock() -> Callable[[], str]:
    ticks = count(1)
    return lambda: f"2026-11-10T10:00:{next(ticks):02d}+00:00"


def test_promotion_rejection_rollback_and_manual_promotion(world: RegistryWorld) -> None:
    v1, v2, v3 = (world.add_version(rate) for rate in (0.6, 0.3, 0.7))
    promoter, registry = Promoter(world.registry, clock()), world.registry

    first = promoter.apply(v1, verdict(True), campaign_run_id="c1")
    assert (first.kind, first.production_before, first.production_after) == (EventKind.PROMOTION, None, v1)
    assert registry.aliases() == {"production": v1}
    assert world.tags(v1)["lifecycle"] == "production" and world.tags(v1)["decided_by"] == "pipeline"
    assert world.tags(v1)["gate_status"] == "passed" and world.tags(v1)["replaced_version"] == ""

    rejected = promoter.apply(v2, verdict(False, compared_to=v1, delta=0.02))
    assert rejected.kind is EventKind.REJECTION and rejected.production_after == v1
    assert registry.aliases() == {"production": v1, "staging": v2}
    tags = world.tags(v2)
    assert (tags["lifecycle"], tags["gate_status"], tags["compared_to_version"]) == ("rejected", "failed", v1)
    assert tags["delta_log_loss"] == "+0.02000" and "Elo pur" in tags["gate_reason"]

    promoter.apply(v3, verdict(True, compared_to=v1, delta=-0.01))
    assert registry.aliases() == {"production": v3, "previous": v1}  # staging (v2, plus ancienne) retiré
    assert (world.tags(v1)["lifecycle"], world.tags(v1)["replaced_by"]) == ("archived", v3)
    assert world.tags(v3)["replaced_version"] == v1

    back = Rollback(registry, clock()).run("probabilités aberrantes sur la terre battue")
    assert (back.kind, back.production_before, back.production_after) == (EventKind.ROLLBACK, v3, v1)
    assert back.title == f"rollback v{v3} → v{v1}"
    assert registry.aliases() == {"production": v1}  # v1 n'avait rien remplacé : plus de `previous`
    bad, good = world.tags(v3), world.tags(v1)
    assert (bad["lifecycle"], bad["rollback_reason"], bad["rolled_back_to"]) == (
        "rolled_back", "probabilités aberrantes sur la terre battue", v1,
    )  # fmt: skip
    assert (good["lifecycle"], good["restored_from"]) == ("production", v3)
    with pytest.raises(RollbackError, match="aucune version `previous`"):
        Rollback(registry).run("encore")

    manual = promoter.promote_manually(v2, "meilleur en shadow sur 3 semaines")
    assert (manual.kind, manual.decided_by.value, manual.production_before) == (
        EventKind.MANUAL_PROMOTION,
        "manual",
        v1,
    )
    assert registry.aliases() == {"production": v2, "previous": v1}
    tags = world.tags(v2)
    assert (tags["lifecycle"], tags["decided_by"], tags["gate_status"]) == ("production", "manual", "failed")
    assert tags["promotion_reason"] == "meilleur en shadow sur 3 semaines"

    with pytest.raises(PromotionError, match="déjà en production"):
        promoter.promote_manually(v2, "deux fois")
    with pytest.raises(PromotionError, match="raison"):
        promoter.promote_manually(v1, "  ")
    with pytest.raises(UnknownVersionError, match="v99"):
        promoter.promote_manually("99", "inconnue")


def test_successive_rollbacks_walk_back_the_history(world: RegistryWorld) -> None:
    v1, v2, v3 = (world.add_version(rate) for rate in (0.6, 0.65, 0.7))
    promoter = Promoter(world.registry)
    for version in (v1, v2, v3):
        promoter.apply(version, verdict(True))
    assert world.registry.aliases() == {"production": v3, "previous": v2}

    Rollback(world.registry).run("incident 1")
    assert world.registry.aliases() == {"production": v2, "previous": v1}
    Rollback(world.registry).run("incident 2")
    assert world.registry.aliases() == {"production": v1}
    assert [world.tags(v)["lifecycle"] for v in (v1, v2, v3)] == ["production", "rolled_back", "rolled_back"]


def test_rollback_needs_a_production_and_a_reason(world: RegistryWorld) -> None:
    with pytest.raises(RollbackError, match="raison"):
        Rollback(world.registry).run("")
    with pytest.raises(RollbackError, match="aucune version de atp-registre en production"):
        Rollback(world.registry).run("panne")
    assert world.registry.aliases() == {}
    assert world.registry.production() is None


def test_every_decision_is_journaled(world: RegistryWorld) -> None:
    v1, v2 = world.add_version(0.6), world.add_version(0.3)
    promoter = Promoter(world.registry)
    world.journal.record(promoter.apply(v1, verdict(True), campaign_run_id="campagne-1"))
    world.journal.record(promoter.apply(v2, verdict(False, compared_to=v1, delta=0.03)))
    world.journal.record(promoter.promote_manually(v2, "test A/B concluant"))
    world.journal.record(Rollback(world.registry).run("plaintes des utilisateurs"))

    runs = world.journal_runs()
    assert [r["event"] for r in runs] == ["promotion", "rejet", "promotion manuelle", "rollback"]
    assert [r["mlflow.runName"] for r in runs] == [f"promotion v{v1}", f"rejet v{v2}", f"promotion manuelle v{v2}",
                                                   f"rollback v{v2} → v{v1}"]  # fmt: skip
    assert runs[0]["campaign_run_id"] == "campagne-1" and runs[0]["gate_status"] == "passed"
    assert runs[3]["reason"] == "plaintes des utilisateurs" and runs[3]["decided_by"] == "manual"
    assert "gate_status" not in runs[3]


def test_a_journal_failure_never_hides_an_applied_decision(
    world: RegistryWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    v1 = world.add_version(0.6)
    event = Promoter(world.registry).apply(v1, verdict(True))

    def broken(_: object) -> str:
        raise MlflowException("serveur tombé")

    monkeypatch.setattr(world.journal, "record", broken)
    assert world.journal.try_record(event) == "journal des décisions non écrit : serveur tombé"
    assert world.registry.aliases() == {"production": v1}  # la décision est bien appliquée
