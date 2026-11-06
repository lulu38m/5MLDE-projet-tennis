"""Candidats de la grille et fabrique de pipelines : prétraitement + modèle, journalisés comme **un seul** modèle.

Le prétraitement (imputation, mise à l'échelle, encodage) est dans le pipeline : MLflow le sauvegarde avec le
modèle, et l'API n'a qu'à fournir les features brutes du `FeatureBuilder`.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tennis_mlops.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from tennis_mlops.training.config import GridValue, ModelKind, TrainConfig

SHORT_NAMES = {
    ModelKind.LOGISTIC_REGRESSION: "logreg",
    ModelKind.HIST_GRADIENT_BOOSTING: "hgb",
    ModelKind.RANDOM_FOREST: "rf",
}
PARAM_LABELS = {"learning_rate": "lr", "max_depth": "depth", "max_iter": "iter", "n_estimators": "trees",
                "min_samples_leaf": "leaf"}  # fmt: skip


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """Une combinaison de la grille : famille, hyperparamètres, graine."""

    kind: ModelKind
    params: Mapping[str, GridValue] = field(default_factory=dict)
    seed: int = 42

    @property
    def name(self) -> str:
        """Nom lisible et stable, utilisé comme nom de run : `hgb lr=0.1 depth=3 iter=300`."""
        parts = [f"{PARAM_LABELS.get(k, k)}={v}" for k, v in sorted(self.params.items())]
        return " ".join([SHORT_NAMES[self.kind], *parts])


def expand_grid(config: TrainConfig) -> list[CandidateSpec]:
    """Produit cartésien de chaque grille, dans l'ordre du fichier de configuration."""
    specs: list[CandidateSpec] = []
    for family in config.candidates:
        names = sorted(family.grid)
        for values in itertools.product(*(family.grid[n] for n in names)):
            specs.append(CandidateSpec(kind=family.model, params=dict(zip(names, values, strict=True)),
                                       seed=config.seed))  # fmt: skip
    return specs


class PipelineFactory:
    """Construit le pipeline complet d'un candidat."""

    def create(self, spec: CandidateSpec) -> Pipeline:
        """`ColumnTransformer` (numérique + catégoriel) suivi du modèle de la famille."""
        return Pipeline([("preprocess", self._preprocess(spec.kind)), ("model", self._model(spec))])

    @staticmethod
    def _preprocess(kind: ModelKind) -> ColumnTransformer:
        numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if kind is ModelKind.LOGISTIC_REGRESSION:  # les arbres n'ont pas besoin de mise à l'échelle
            numeric_steps.append(("scale", StandardScaler()))
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="inconnu")),
                ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            [
                ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
                ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ],
            verbose_feature_names_out=False,
        )

    @staticmethod
    def _model(spec: CandidateSpec) -> ClassifierMixin:
        params = dict(spec.params)
        if spec.kind is ModelKind.LOGISTIC_REGRESSION:
            return LogisticRegression(max_iter=2000, random_state=spec.seed, **params)
        if spec.kind is ModelKind.HIST_GRADIENT_BOOSTING:
            return HistGradientBoostingClassifier(random_state=spec.seed, **params)
        # n_jobs=1 : le parallélisme se fait entre candidats (Prefect), pas à l'intérieur d'un candidat.
        return RandomForestClassifier(random_state=spec.seed, n_jobs=1, **params)
