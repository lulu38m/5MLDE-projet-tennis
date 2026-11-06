"""Registre MLflow : chaque campagne ajoute une **version** du modèle retenu sous un nom stable.

Une version enregistrée n'est pas une version en service : elle porte `lifecycle=candidate` et
`qualification=pending`. La décision de mise en service (alias `production`, règles, rollback) est la phase 5.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from tennis_mlops.training.tracking import MlflowTracker

PRODUCTION_ALIAS = "production"
MODEL_DESCRIPTION = (
    "Probabilité que le joueur 1 gagne un match ATP en simple. Entrée : les features brutes du FeatureBuilder "
    "(tennis_mlops.features) ; sortie : predict_proba, colonne 1 = le joueur 1 gagne."
)


@dataclass(frozen=True)
class RegisteredVersion:
    """Une version créée dans le registre."""

    name: str
    version: str
    model_uri: str
    run_id: str


@dataclass(frozen=True)
class ProductionModel:
    """La version actuellement en service (alias `production`) et son modèle chargé."""

    version: ModelVersion
    model: Any


class Registrar:
    """Enregistre les versions et retrouve celle en production."""

    def __init__(self, tracker: MlflowTracker) -> None:
        self.tracker = tracker

    def register(
        self, name: str, model_uri: str, run_id: str, tags: Mapping[str, str], description: str
    ) -> RegisteredVersion:
        """Nouvelle version sous `name`, reliée à son run et à son URI de modèle, avec ses tags."""
        client = self.tracker.client
        version = mlflow.register_model(model_uri, name, tags=dict(tags))
        client.update_model_version(name, version.version, description=description)
        if not client.get_registered_model(name).description:
            client.update_registered_model(name, description=MODEL_DESCRIPTION)
        return RegisteredVersion(name=name, version=str(version.version), model_uri=model_uri, run_id=run_id)

    def production(self, name: str) -> ProductionModel | None:
        """Version désignée par l'alias `production`, ou None s'il n'existe pas encore (phase 5)."""
        try:
            version = self.tracker.client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
        except MlflowException:
            return None
        model = mlflow.sklearn.load_model(f"models:/{name}@{PRODUCTION_ALIAS}")
        return ProductionModel(version=version, model=model)
