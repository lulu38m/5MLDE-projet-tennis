"""Configuration d'une campagne d'entraînement (`configs/train.yaml`), validée avant tout calcul."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tennis_mlops.config import PROJECT_ROOT
from tennis_mlops.features import FeatureConfig

DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"

GridValue = int | float | str | bool | None


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelKind(StrEnum):
    """Familles de modèles disponibles dans la grille."""

    LOGISTIC_REGRESSION = "logistic_regression"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
    RANDOM_FOREST = "random_forest"


class SplitConfig(_Strict):
    """Découpage temporel : entraînement, puis validation (choix), puis test (bilan), sans chevauchement."""

    train_start: date
    validation_months: int = Field(ge=1, le=24)
    test_months: int = Field(ge=1, le=24)
    min_rows: int = Field(default=300, ge=10)


class CandidateGrid(_Strict):
    """Une famille de modèles et ses hyperparamètres à croiser."""

    model: ModelKind
    grid: dict[str, tuple[GridValue, ...]] = {}

    @field_validator("grid")
    @classmethod
    def _non_empty_values(cls, grid: dict[str, tuple[GridValue, ...]]) -> dict[str, tuple[GridValue, ...]]:
        empty = [name for name, values in grid.items() if not values]
        if empty:
            msg = f"hyperparamètre(s) sans valeur : {', '.join(empty)}"
            raise ValueError(msg)
        return grid


class EvaluationConfig(_Strict):
    """Ce que l'évaluation calcule en plus des métriques globales."""

    calibration_bins: int = Field(default=10, ge=2, le=50)
    permutation_rows: int = Field(default=2000, ge=100)
    permutation_repeats: int = Field(default=5, ge=1, le=50)
    latency_calls: int = Field(default=200, ge=10)
    latency_batch_rows: int = Field(default=1000, ge=10)
    worst_predictions: int = Field(default=50, ge=1)
    min_segment_rows: int = Field(default=30, ge=1)
    rank_gap_edges: tuple[int, ...] = (10, 50, 200)

    @field_validator("rank_gap_edges")
    @classmethod
    def _increasing(cls, edges: tuple[int, ...]) -> tuple[int, ...]:
        if list(edges) != sorted(set(edges)) or (edges and edges[0] <= 0):
            msg = "rank_gap_edges : bornes strictement croissantes et positives attendues"
            raise ValueError(msg)
        return edges


class TrainConfig(_Strict):
    """Une campagne : données, features, grille, sélection, évaluation."""

    experiment: str = Field(min_length=1)
    registered_model: str = Field(min_length=1, pattern=r"^[\w-]+$")
    seed: int = 42
    split: SplitConfig
    features: FeatureConfig = FeatureConfig()
    selection_metric: str = Field(default="log_loss", pattern=r"^(log_loss|brier)$")
    workers: int = Field(default=4, ge=1, le=32)
    candidates: tuple[CandidateGrid, ...] = Field(min_length=1)
    evaluation: EvaluationConfig = EvaluationConfig()

    @model_validator(mode="after")
    def _unique_families(self) -> Self:
        kinds = [c.model for c in self.candidates]
        if len(kinds) != len(set(kinds)):
            msg = "une famille de modèles ne doit apparaître qu'une fois (croiser ses valeurs dans `grid`)"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_TRAIN_CONFIG_PATH) -> Self:
        """Charge et valide `configs/train.yaml`."""
        with path.open(encoding="utf-8") as handle:
            raw: object = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            msg = f"{path} doit contenir un objet YAML à la racine"
            raise TypeError(msg)
        return cls.model_validate(raw)
