import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from tennis_mlops.training.config import ModelKind
from tennis_mlops.training.drift import drift, profile
from tennis_mlops.training.evaluate import (
    SEGMENT_DIMENSIONS,
    Evaluator,
    calibration_table,
    classification_metrics,
    elo_baseline,
    expected_calibration_error,
    rank_baseline,
    segment_labels,
)
from tennis_mlops.training.pipeline import CandidateSpec, PipelineFactory
from tennis_mlops.training.report import UntrustedTypeError, metric_key, model_contract, trusted_types
from tennis_mlops.training.split import Split
from tennis_mlops.training.trainer import Trainer
from tests.support.training import small_config


@pytest.mark.parametrize("kind", list(ModelKind))
def test_every_family_trains_and_beats_chance(kind: ModelKind, split: Split) -> None:
    params = {ModelKind.RANDOM_FOREST: {"n_estimators": 20, "max_depth": 4}}.get(kind, {})
    spec = CandidateSpec(kind, params)
    fitted = Trainer().fit(spec, PipelineFactory().create(spec), split)
    evaluation = Evaluator(small_config().evaluation).evaluate(spec.name, fitted.pipeline, split)
    assert fitted.fit_seconds > 0
    assert evaluation.metrics["test"]["roc_auc"] > 0.6
    assert evaluation.metrics["test"]["log_loss"] < math.log(2)
    assert set(evaluation.flat_metrics()) >= {"train/log_loss", "validation/ece", "test/accuracy"}
    assert "baseline_x/test_log_loss" in evaluation.flat_metrics("baseline_x")
    assert all(t.startswith(("sklearn.", "numpy.dtype")) for t in trusted_types(fitted.pipeline))


def test_pipeline_accepts_missing_values_and_unknown_categories(split: Split) -> None:
    spec = CandidateSpec(ModelKind.LOGISTIC_REGRESSION, {"C": 1.0})
    fitted = Trainer().fit(spec, PipelineFactory().create(spec), split)
    odd = Split.features(split.test).head(3).copy()
    odd["surface"] = "Moquette"
    odd["p1_hand"] = None
    odd["rank_log_ratio"] = np.nan
    proba = fitted.pipeline.predict_proba(odd)[:, 1]
    assert ((proba > 0) & (proba < 1)).all()
    signature, example = model_contract(fitted.pipeline, Split.features(split.validation))
    assert len(example) == 5
    assert signature.inputs.input_names()[0] == "rank_log_ratio"


def test_foreign_types_are_never_trusted() -> None:
    class Homemade(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
        pass

    with pytest.raises(UntrustedTypeError, match="Homemade"):
        trusted_types(Homemade())


def test_metrics_on_known_values() -> None:
    labels = np.array([1, 0, 1, 0])
    perfect = classification_metrics(labels, np.array([1.0, 0.0, 1.0, 0.0]), bins=10)
    assert perfect["accuracy"] == 1.0
    assert perfect["roc_auc"] == 1.0
    assert perfect["log_loss"] < 1e-5
    coin = classification_metrics(labels, np.full(4, 0.5), bins=10)
    assert coin["log_loss"] == pytest.approx(math.log(2))
    assert coin["brier"] == pytest.approx(0.25)
    assert coin["ece"] == pytest.approx(0.0)
    assert math.isnan(classification_metrics(np.ones(3, dtype=int), np.full(3, 0.7), bins=10)["roc_auc"])


def test_calibration_error_measures_overconfidence() -> None:
    labels = np.array([1] * 6 + [0] * 4)  # 60 % de victoires…
    probabilities = np.full(10, 0.9)  # …annoncées à 90 %
    assert expected_calibration_error(labels, probabilities, bins=10) == pytest.approx(0.3)
    table = calibration_table(labels, probabilities, bins=10)
    assert table[["rows", "observed_rate"]].iloc[0].tolist() == [10, 0.6]


def test_segment_labels() -> None:
    frame = pd.DataFrame({"surface": ["Hard", None], "tourney_level": ["G", "A"], "round": ["F", None],
                          "p1_rank": [1.0, 300.0], "p2_rank": [5.0, None]})  # fmt: skip
    assert segment_labels(frame, "surface", (10,)).tolist() == ["Hard", "inconnue"]
    assert segment_labels(frame, "niveau", (10,)).tolist() == ["Grand Chelem", "ATP 250/500"]
    assert segment_labels(frame, "grand_chelem", (10,)).tolist() == ["Grand Chelem", "autres tournois"]
    assert segment_labels(frame, "ecart_classement", (10, 50)).tolist() == ["0-10", "inconnu"]
    with pytest.raises(ValueError, match="inconnue"):
        segment_labels(frame, "météo", (10,))


def test_segments_costs_importance_and_worst_predictions(split: Split) -> None:
    config = small_config()
    spec = CandidateSpec(ModelKind.LOGISTIC_REGRESSION, {"C": 1.0})
    fitted = Trainer().fit(spec, PipelineFactory().create(spec), split)
    evaluator = Evaluator(config.evaluation)
    evaluation = evaluator.evaluate(spec.name, fitted.pipeline, split)
    for dimension in SEGMENT_DIMENSIONS:
        table = evaluator.segments(evaluation, split, dimension)
        assert table["rows"].sum() == len(split.test)
    gaps = evaluator.segments(evaluation, split, "ecart_classement")["segment"].tolist()
    assert gaps == sorted(gaps, key=lambda g: int(g.split("-")[0].rstrip("+")))
    costs = evaluator.costs(fitted.pipeline, split)
    assert costs["cost/latency_1_row_p95_ms"] >= costs["cost/latency_1_row_p50_ms"] > 0
    assert costs["cost/model_size_kb"] > 0
    importance = evaluator.feature_importance(fitted.pipeline, split, seed=1)
    assert importance["importance"].iloc[0] == importance["importance"].max()
    worst = evaluator.worst_predictions(evaluation, split)
    assert len(worst) == config.evaluation.worst_predictions
    assert worst["loss"].is_monotonic_decreasing
    assert (worst["predicted_winner"] != worst["actual_winner"]).all()


def test_baselines(split: Split) -> None:
    evaluator = Evaluator(small_config().evaluation)
    elo = evaluator.evaluate_probabilities("elo", split, elo_baseline)
    rank = evaluator.evaluate_probabilities("rang", split, rank_baseline(split.train))
    assert elo.metrics["test"]["roc_auc"] > 0.55
    assert rank.metrics["test"]["roc_auc"] > 0.55


def test_drift_is_zero_on_the_same_data_and_high_on_shifted_data(split: Split) -> None:
    reference = profile(split.train)
    assert drift(reference, split.train)["max_psi"] == pytest.approx(0.0, abs=1e-9)
    shifted = split.train.copy()
    shifted["elo_diff"] = shifted["elo_diff"] + 500
    shifted["surface"] = "Grass"
    report = drift(reference, shifted)
    assert report["features_over_alert"] == ["elo_diff", "surface"]
    assert drift({}, shifted)["max_psi"] == 0.0


def test_metric_keys_are_accepted_by_mlflow() -> None:
    assert metric_key("segments", "ecart_classement", "200+") == "segments/ecart_classement.200plus"
    assert metric_key("segments", "niveau", "ATP 250/500") == "segments/niveau.ATP 250-500"


def test_monthly_performance_and_learning_curves(split: Split) -> None:
    evaluator = Evaluator(small_config().evaluation)
    spec = CandidateSpec(ModelKind.HIST_GRADIENT_BOOSTING, {"max_iter": 20, "max_depth": 3})
    fitted = Trainer().fit(spec, PipelineFactory().create(spec), split)
    evaluation = evaluator.evaluate(spec.name, fitted.pipeline, split)
    elo = evaluator.evaluate_probabilities("elo", split, elo_baseline)

    monthly = evaluator.monthly(evaluation, elo, split)
    assert monthly["matches"].sum() <= len(split.validation) + len(split.test)
    assert (monthly["matches"] >= small_config().evaluation.min_segment_rows).all()  # mois partiels écartés
    assert list(monthly["period"].unique()) == ["validation", "test"]
    assert monthly["month"].is_monotonic_increasing
    assert np.allclose(monthly["gap_vs_reference"], monthly["log_loss"] - monthly["reference_log_loss"])

    curve = Evaluator.learning_curve(fitted.pipeline, split)
    assert curve is not None
    assert list(curve["iteration"]) == list(range(1, 21))
    assert curve["train_log_loss"].iloc[-1] < curve["train_log_loss"].iloc[0]
    logreg = CandidateSpec(ModelKind.LOGISTIC_REGRESSION, {"C": 1.0})
    assert (
        Evaluator.learning_curve(Trainer().fit(logreg, PipelineFactory().create(logreg), split).pipeline, split) is None
    )
