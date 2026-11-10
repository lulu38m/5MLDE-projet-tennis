"""Le chargeur suit l'alias `production` : promotion et rollback sont servis sans redémarrage."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
from mlflow.exceptions import MlflowException

from tennis_mlops.registry.loader import AliasModelLoader
from tennis_mlops.registry.model_registry import ModelRegistry
from tennis_mlops.registry.promotion import Promoter
from tennis_mlops.registry.rollback import Rollback
from tests.support.registry import RegistryWorld, probe, registry_world, verdict


@pytest.fixture
def world(tmp_path: Path) -> Iterator[RegistryWorld]:
    with registry_world(tmp_path) as w:
        yield w


def test_promotion_and_rollback_are_served_without_restart(world: RegistryWorld) -> None:
    v1, v2 = world.add_version(0.6), world.add_version(0.3)
    elsewhere = f"sqlite:///{world.folder / 'autre.db'}"  # un nouveau processus n'a pas les URI de la campagne
    mlflow.set_tracking_uri(elsewhere)
    mlflow.set_registry_uri(elsewhere)
    loader = AliasModelLoader(ModelRegistry.connect(world.uri, world.registry.name))  # comme l'API
    assert loader.refresh().error == "alias `production` posé sur aucune version"
    assert loader.current is None

    Promoter(world.registry).apply(v1, verdict(True))
    first = loader.refresh()
    assert (first.changed, first.version, first.previous_version) == (True, v1, None)
    assert loader.current is not None and probe(loader.current.model) == pytest.approx(0.6)
    assert loader.current.tags["lifecycle"] == "production"
    unchanged = loader.refresh()
    assert (unchanged.changed, unchanged.version, unchanged.stale) == (False, v1, False)

    Promoter(world.registry).apply(v2, verdict(True))
    swapped = loader.refresh()
    assert (swapped.changed, swapped.version, swapped.previous_version) == (True, v2, v1)
    assert probe(loader.current.model) == pytest.approx(0.3)

    Rollback(world.registry).run("v2 annonce 30 % pour tout le monde")
    restored = loader.refresh()  # même objet, même processus : aucun redéploiement
    assert (restored.changed, restored.version, restored.previous_version) == (True, v1, v2)
    assert probe(loader.current.model) == pytest.approx(0.6)


class Unreachable:
    """Registre dont le serveur ne répond plus."""

    name = "atp-registre"

    def by_alias(self, alias: object) -> None:
        raise MlflowException("Connection refused")


def test_the_last_model_is_kept_when_mlflow_is_down(world: RegistryWorld) -> None:
    v1 = world.add_version(0.6)
    Promoter(world.registry).apply(v1, verdict(True))
    loader = AliasModelLoader(world.registry)
    loader.refresh()
    loader.registry = Unreachable()  # type: ignore[assignment]
    outcome = loader.refresh()
    assert outcome.stale and not outcome.changed and outcome.version == v1
    assert "modèle conservé" in (outcome.error or "")
    assert loader.current is not None and probe(loader.current.model) == pytest.approx(0.6)
