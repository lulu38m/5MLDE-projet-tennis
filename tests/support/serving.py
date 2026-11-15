"""Un service de prédiction complet pour les tests : vraie base SQLite, vrai historique, vrai pipeline scikit-learn.

Seul le registre MLflow est remplacé (`FakeRegistry`) : on y pose l'alias à la main, et on peut le rendre injoignable
pour rejouer le scénario S8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from tennis_mlops.api.app import create_app
from tennis_mlops.config import ApiConfig
from tennis_mlops.features import FEATURE_COLUMNS, FEATURES_VERSION, LABEL, FeatureBuilder, FeatureConfig
from tennis_mlops.ingestion.store import IngestionStore, matches
from tennis_mlops.registry.loader import AliasModelLoader
from tennis_mlops.serving.features import FeatureStore
from tennis_mlops.serving.model import ModelHolder
from tennis_mlops.serving.persistence import ApiStore, AsyncWriter
from tennis_mlops.serving.service import PredictionService
from tennis_mlops.training.config import ModelKind
from tennis_mlops.training.pipeline import CandidateSpec, PipelineFactory
from tests.support.synthetic import synthetic_matches


def train_pipeline(frame: pd.DataFrame, c: float = 1.0) -> Pipeline:
    """Un pipeline logistique entraîné sur les features de ces matchs."""
    features = FeatureBuilder().build(frame)
    pipeline = PipelineFactory().create(CandidateSpec(ModelKind.LOGISTIC_REGRESSION, {"C": c}))
    pipeline.fit(features[list(FEATURE_COLUMNS)], features[LABEL])
    return pipeline


@dataclass
class FakeRegistry:
    """Le strict nécessaire de `ModelRegistry` pour `AliasModelLoader` : un alias, des versions, une panne."""

    name: str = "atp-winner"
    models: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, dict[str, str]] = field(default_factory=dict)
    production: str | None = None
    down: bool = False

    def by_alias(self, _: object) -> SimpleNamespace | None:
        if self.down:
            msg = "Connection refused"
            raise MlflowException(msg)
        if self.production is None:
            return None
        return SimpleNamespace(
            version=self.production, run_id=f"run-{self.production}", tags=self.tags[self.production]
        )

    def load(self, version: str) -> Any:
        return self.models[version]

    def add(self, version: str, model: Any, features_version: str = FEATURES_VERSION) -> None:
        self.models[version] = model
        self.tags[version] = {"features_version": features_version, "lifecycle": "production", "candidate": "logreg"}


@dataclass
class ServingWorld:
    """Base, service, application et client HTTP."""

    path: Path
    frame: pd.DataFrame
    ingestion: IngestionStore
    store: ApiStore
    registry: FakeRegistry
    service: PredictionService
    app: FastAPI
    client: TestClient

    def ready(self) -> ServingWorld:
        """Premier chargement (ce que fait le thread de rafraîchissement au démarrage)."""
        self.service.features.refresh()
        self.service.model.refresh()
        return self

    @property
    def last_day(self) -> date:
        """Dernière journée de l'historique servi."""
        snapshot = self.service.features.current
        assert snapshot is not None and snapshot.last_day is not None
        return snapshot.last_day

    def flush(self) -> None:
        """Écrit les traces en attente."""
        self.service.writer.flush()

    def rows(self, sql: str) -> list[dict[str, Any]]:
        """Lecture SQL brute."""
        with self.store.engine.connect() as conn:
            return [dict(r) for r in conn.execute(sa.text(sql)).mappings()]

    def add_matches(self, frame: pd.DataFrame) -> None:
        """Ajoute des matchs en base (un nouveau lot)."""
        insert_matches(self.ingestion, frame)


def insert_matches(store: IngestionStore, frame: pd.DataFrame) -> None:
    """Insère un DataFrame au schéma `matches`."""
    columns = set(matches.columns.keys())
    records = [{k: (None if pd.isna(v) else v) for k, v in row.items() if k in columns}
               for row in frame.to_dict("records")]  # fmt: skip
    with store.engine.begin() as conn:
        conn.execute(sa.insert(matches), records)


def serving_world(tmp_path: Path, weeks: int = 60, admin_token: str | None = None) -> ServingWorld:
    """Un monde prêt à charger : 60 semaines synthétiques en base, v1 entraînée et désignée par `production`."""
    frame = synthetic_matches(weeks=weeks, matches_per_week=30, players=40, surface_effect=0.5)
    ingestion = IngestionStore.from_url(f"sqlite:///{tmp_path / 'tennis.sqlite'}")
    ingestion.create_schema()
    insert_matches(ingestion, frame)
    store = ApiStore(ingestion.engine)
    store.create_schema()
    registry = FakeRegistry()
    registry.add("1", train_pipeline(frame))
    registry.production = "1"
    service = PredictionService(
        features=FeatureStore(ingestion, FeatureConfig()),
        model=ModelHolder(AliasModelLoader(registry, load=registry.load)),  # type: ignore[arg-type]
        store=store,
        writer=AsyncWriter(store),
        config=ApiConfig(max_batch=50),
    )
    app = create_app(service, admin_token=admin_token, manage_service=False)
    return ServingWorld(tmp_path, frame, ingestion, store, registry, service, app, TestClient(app))


def match_body(p1: int = 100_001, p2: int = 100_002, **extra: object) -> dict[str, Any]:
    """Corps de `POST /predict` minimal."""
    return {"surface": "Clay", "tourney_level": "M", "round": "QF", "best_of": 3,
            "player1": {"id": p1}, "player2": {"id": p2}, **extra}  # fmt: skip
