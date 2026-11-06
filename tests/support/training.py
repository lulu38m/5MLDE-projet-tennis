"""Configuration d'entraînement minuscule, partagée par les tests de l'entraînement et du flow."""

from __future__ import annotations

from datetime import date

from tennis_mlops.training.config import CandidateGrid, EvaluationConfig, ModelKind, SplitConfig, TrainConfig


def small_config(**overrides: object) -> TrainConfig:
    """Grille minuscule et évaluation légère : une campagne complète en quelques secondes."""
    values: dict[str, object] = {
        "experiment": "tests",
        "registered_model": "atp-tests",
        "split": SplitConfig(train_start=date(2010, 6, 1), validation_months=6, test_months=6, min_rows=100),
        "candidates": (
            CandidateGrid(model=ModelKind.LOGISTIC_REGRESSION, grid={"C": (0.1, 1.0)}),
            CandidateGrid(model=ModelKind.HIST_GRADIENT_BOOSTING, grid={"max_iter": (30,), "max_depth": (3,)}),
        ),
        "workers": 2,
        "evaluation": EvaluationConfig(permutation_rows=200, permutation_repeats=2, latency_calls=20,
                                       latency_batch_rows=50, worst_predictions=5),
    }  # fmt: skip
    values.update(overrides)
    return TrainConfig.model_validate(values)
