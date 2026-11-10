"""Accès au registre MLflow pour UN modèle (`atp-winner`) : versions, alias, tags, chargement.

Module technique, sans règle métier : la décision (promotion, rejet, rollback) est dans `promotion.py` et
`rollback.py`. Chaque campagne ajoute une **version** ; une version n'est servie que si l'alias `production` la
désigne.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from tennis_mlops.registry.tagging import Alias, clean

MODEL_DESCRIPTION = (
    "Probabilité que le joueur 1 gagne un match ATP en simple. Entrée : les features brutes du FeatureBuilder "
    "(tennis_mlops.features) ; sortie : predict_proba, colonne 1 = le joueur 1 gagne. L'API charge "
    "models:/atp-winner@production ; stratégie de tagging : docs/TAGGING.md."
)


class UnknownVersionError(LookupError):
    """La version demandée n'existe pas dans le registre."""


@dataclass(frozen=True)
class RegisteredVersion:
    """Une version créée dans le registre."""

    name: str
    version: str
    model_uri: str
    run_id: str


@dataclass(frozen=True)
class ProductionModel:
    """La version désignée par l'alias `production` et son modèle chargé."""

    version: ModelVersion
    model: Any


class ModelRegistry:
    """Les versions d'un modèle enregistré, ses alias et ses tags."""

    def __init__(self, client: MlflowClient, name: str) -> None:
        self.client = client
        self.name = name

    @classmethod
    def connect(cls, tracking_uri: str, name: str) -> ModelRegistry:
        """Registre d'un serveur MLflow, pour un processus qui ne parle qu'à ce serveur (l'API, la ligne de commande).

        Le chargement d'un modèle (`models:/…`) passe par l'API globale de MLflow : on y fixe aussi les URI, sinon
        MLflow 3 retombe sur une base SQLite locale `mlflow.db` et ne trouve pas le modèle.
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        return cls(MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri), name)

    # ------------------------------------------------------------------ écritures

    def register(self, model_uri: str, run_id: str, tags: Mapping[str, str], description: str) -> RegisteredVersion:
        """Nouvelle version reliée à son run et à son URI de modèle, avec ses tags."""
        created = mlflow.register_model(model_uri, self.name, tags=clean(tags))
        self.client.update_model_version(self.name, created.version, description=description)
        if not self.client.get_registered_model(self.name).description:
            self.client.update_registered_model(self.name, description=MODEL_DESCRIPTION)
        return RegisteredVersion(name=self.name, version=str(created.version), model_uri=model_uri, run_id=run_id)

    def tag(self, version: str, tags: Mapping[str, object]) -> None:
        """Pose (ou remplace) des tags sur une version."""
        for key, value in clean(tags).items():
            self.client.set_model_version_tag(self.name, version, key, value)

    def set_alias(self, alias: Alias, version: str) -> None:
        """Fait désigner `version` par l'alias (il quitte la version précédente)."""
        self.client.set_registered_model_alias(self.name, alias.value, version)

    def drop_alias(self, alias: Alias) -> None:
        """Retire l'alias s'il existe."""
        if alias.value in self.aliases():
            self.client.delete_registered_model_alias(self.name, alias.value)

    # ------------------------------------------------------------------ lectures

    def get(self, version: str) -> ModelVersion:
        """Une version par son numéro."""
        try:
            return self.client.get_model_version(self.name, version)
        except MlflowException as exc:
            msg = f"{self.name} v{version} n'existe pas dans le registre"
            raise UnknownVersionError(msg) from exc

    def by_alias(self, alias: Alias) -> ModelVersion | None:
        """La version désignée par l'alias, ou None s'il n'est posé nulle part."""
        try:
            return self.client.get_model_version_by_alias(self.name, alias.value)
        except MlflowException as exc:
            if exc.error_code in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
                return None
            raise

    def aliases(self) -> dict[str, str]:
        """Alias → numéro de version (vide si le modèle n'est pas encore enregistré)."""
        try:
            return {k: str(v) for k, v in self.client.get_registered_model(self.name).aliases.items()}
        except MlflowException as exc:
            if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
                return {}
            raise

    def versions(self) -> list[ModelVersion]:
        """Toutes les versions, de la plus récente à la plus ancienne."""
        found = self.client.search_model_versions(f"name = '{self.name}'")
        return sorted(found, key=lambda v: int(v.version), reverse=True)

    def load(self, version: str) -> Pipeline:
        """Le pipeline scikit-learn d'une version (par son numéro : insensible à un alias qui bouge entre-temps)."""
        return mlflow.sklearn.load_model(f"models:/{self.name}/{version}")

    def production(self) -> ProductionModel | None:
        """La version en service et son modèle, ou None tant qu'aucune version n'a été promue."""
        version = self.by_alias(Alias.PRODUCTION)
        if version is None:
            return None
        return ProductionModel(version=version, model=self.load(str(version.version)))
