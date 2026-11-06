"""Évaluation d'un candidat : métriques globales, par segment, coûts, importance des features, erreurs.

Toutes les métriques portent sur une **probabilité** (le joueur 1 gagne) : log loss et Brier jugent la qualité de
la probabilité, l'ECE sa calibration (« 70 % » doit gagner 7 fois sur 10), accuracy et ROC AUC le classement.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import skops.io
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from tennis_mlops.features import FEATURE_COLUMNS
from tennis_mlops.training.config import EvaluationConfig
from tennis_mlops.training.split import PARTS, Split

Array = npt.NDArray[Any]

EPSILON = 1e-6
SEGMENT_DIMENSIONS = ("surface", "niveau", "tour", "ecart_classement", "grand_chelem")
LEVEL_NAMES = {"G": "Grand Chelem", "M": "Masters 1000", "A": "ATP 250/500", "F": "Finales", "D": "Coupe Davis",
               "O": "Jeux olympiques"}  # fmt: skip


class ProbabilityModel(Protocol):
    """Ce que l'évaluation attend d'un modèle : une probabilité par classe."""

    def predict_proba(self, features: pd.DataFrame) -> Array:
        """Une colonne par classe ; la colonne 1 est la probabilité que le joueur 1 gagne."""
        ...


@dataclass
class Evaluation:
    """Probabilités et métriques d'un modèle sur chaque partition."""

    name: str
    probabilities: dict[str, Array]
    metrics: dict[str, dict[str, float]]
    extra: dict[str, Any] = field(default_factory=dict)

    def flat_metrics(self, section: str | None = None) -> dict[str, float]:
        """Noms MLflow rangés par section : `test/log_loss`, ou `baseline_elo/test_log_loss` avec une section.

        Le `/` crée une section repliable dans l'onglet « Model metrics » de MLflow.
        """
        return {
            (f"{part}/{name}" if section is None else f"{section}/{part}_{name}"): value
            for part, values in self.metrics.items()
            for name, value in values.items()
        }


# --------------------------------------------------------------------------- métriques


def expected_calibration_error(labels: Array, probabilities: Array, bins: int) -> float:
    """Écart moyen, pondéré par l'effectif, entre probabilité annoncée et fréquence observée."""
    table = calibration_table(labels, probabilities, bins)
    if table.empty:
        return math.nan
    weights = table["rows"] / table["rows"].sum()
    return float((weights * (table["mean_probability"] - table["observed_rate"]).abs()).sum())


def calibration_table(labels: Array, probabilities: Array, bins: int) -> pd.DataFrame:
    """Par tranche de probabilité : probabilité moyenne annoncée, taux de victoire observé, effectif."""
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    frame = pd.DataFrame({"bin": index, "p": probabilities, "y": labels})
    grouped = frame.groupby("bin").agg(mean_probability=("p", "mean"), observed_rate=("y", "mean"), rows=("y", "size"))
    grouped["bin_start"] = edges[grouped.index]
    return grouped.reset_index(drop=True)


def classification_metrics(labels: Array, probabilities: Array, bins: int) -> dict[str, float]:
    """Métriques d'une probabilité binaire ; ROC AUC à NaN si une seule classe est présente."""
    p = np.clip(probabilities, EPSILON, 1 - EPSILON)
    both_classes = len(np.unique(labels)) == 2
    return {
        "log_loss": float(log_loss(labels, p, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, p)),
        "accuracy": float(accuracy_score(labels, p >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, p)) if both_classes else math.nan,
        "ece": expected_calibration_error(labels, p, bins),
        "rows": float(len(labels)),
    }


# --------------------------------------------------------------------------- segments


def segment_labels(frame: pd.DataFrame, dimension: str, rank_gap_edges: tuple[int, ...]) -> pd.Series[Any]:
    """Le segment de chaque match pour une dimension (surface, niveau, tour, écart de classement, Grand Chelem)."""
    if dimension == "surface":
        return frame["surface"].fillna("inconnue")
    if dimension == "niveau":
        return frame["tourney_level"].map(LEVEL_NAMES).fillna("autre")
    if dimension == "tour":
        return frame["round"].fillna("inconnu")
    if dimension == "grand_chelem":
        return frame["tourney_level"].eq("G").map({True: "Grand Chelem", False: "autres tournois"})
    if dimension == "ecart_classement":
        gap = (frame["p1_rank"] - frame["p2_rank"]).abs()
        names = [f"{lo}-{hi}" for lo, hi in zip((0, *rank_gap_edges), rank_gap_edges, strict=False)]
        bucket = pd.cut(gap, [0, *rank_gap_edges, math.inf], right=False, labels=[*names, f"{rank_gap_edges[-1]}+"])
        return bucket.astype("object").where(gap.notna(), "inconnu")
    msg = f"dimension de segment inconnue : {dimension}"
    raise ValueError(msg)


class Evaluator:
    """Calcule tout ce qui sert à comparer et comprendre un candidat."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def evaluate(self, name: str, model: ProbabilityModel, split: Split) -> Evaluation:
        """Probabilités et métriques sur train, validation et test."""
        probabilities = {part: _positive(model, Split.features(split.part(part))) for part in PARTS}
        return self._evaluation(name, split, probabilities)

    def evaluate_probabilities(self, name: str, split: Split, scorer: Callable[[pd.DataFrame], Array]) -> Evaluation:
        """Idem pour une baseline qui calcule directement une probabilité."""
        return self._evaluation(name, split, {part: scorer(split.part(part)) for part in PARTS})

    def _evaluation(self, name: str, split: Split, probabilities: dict[str, Array]) -> Evaluation:
        metrics = {
            part: classification_metrics(Split.labels(split.part(part)).to_numpy(), probabilities[part],
                                         self.config.calibration_bins)
            for part in PARTS
        }  # fmt: skip
        return Evaluation(name=name, probabilities=probabilities, metrics=metrics)

    def segments(self, evaluation: Evaluation, split: Split, dimension: str, part: str = "test") -> pd.DataFrame:
        """Métriques par segment d'une dimension ; les segments trop petits sont signalés sans être évalués."""
        frame = split.part(part)
        labels = segment_labels(frame, dimension, self.config.rank_gap_edges)
        y, p = Split.labels(frame).to_numpy(), evaluation.probabilities[part]
        rows = []
        for segment in sorted(labels.unique(), key=_segment_order):
            mask = (labels == segment).to_numpy()
            entry: dict[str, Any] = {"dimension": dimension, "segment": segment, "rows": int(mask.sum())}
            if mask.sum() >= self.config.min_segment_rows:
                entry.update({k: v for k, v in classification_metrics(y[mask], p[mask], self.config.calibration_bins)
                              .items() if k != "rows"})  # fmt: skip
            rows.append(entry)
        return pd.DataFrame(rows)

    def costs(self, model: ProbabilityModel, split: Split) -> dict[str, float]:
        """Latence d'inférence (une ligne : p50 / p95 ; un lot) et taille du modèle sérialisé (skops)."""
        cfg = self.config
        features = Split.features(split.test)
        sample = features.sample(n=cfg.latency_calls, replace=len(features) < cfg.latency_calls, random_state=0)
        timings = []
        for i in range(len(sample)):
            started = time.perf_counter()
            model.predict_proba(sample.iloc[[i]])
            timings.append((time.perf_counter() - started) * 1000)
        batch = features.sample(n=cfg.latency_batch_rows, replace=len(features) < cfg.latency_batch_rows,
                                random_state=0)  # fmt: skip
        started = time.perf_counter()
        model.predict_proba(batch)
        batch_ms = (time.perf_counter() - started) * 1000
        quantiles = statistics.quantiles(timings, n=20)
        return {
            "cost/latency_1_row_p50_ms": statistics.median(timings),
            "cost/latency_1_row_p95_ms": quantiles[18],
            f"cost/latency_{cfg.latency_batch_rows}_rows_ms": batch_ms,
            "cost/model_size_kb": len(skops.io.dumps(model)) / 1024,
        }

    def feature_importance(self, model: ProbabilityModel, split: Split, seed: int) -> pd.DataFrame:
        """Importance par permutation sur la validation : hausse du log loss quand une feature est mélangée."""
        cfg = self.config
        frame = split.validation.sample(n=min(cfg.permutation_rows, len(split.validation)), random_state=seed)
        result = permutation_importance(
            model, Split.features(frame), Split.labels(frame), scoring="neg_log_loss",
            n_repeats=cfg.permutation_repeats, random_state=seed, n_jobs=1,
        )  # fmt: skip
        return (
            pd.DataFrame({"feature": list(FEATURE_COLUMNS), "importance": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )  # fmt: skip

    def monthly(self, evaluation: Evaluation, reference: Evaluation, split: Split) -> pd.DataFrame:
        """Performance mois par mois sur la validation puis le test, et écart à une référence (l'Elo pur).

        Une ligne par mois : c'est ce qui montre si le modèle se dégrade avec le temps, ce qu'une moyenne cache.
        Les mois incomplets aux bords des partitions (moins de `min_segment_rows` matchs) sont écartés : sur quelques
        matchs, le log loss ne veut rien dire.
        """
        rows = []
        for part in ("validation", "test"):
            frame = split.part(part)
            months = pd.to_datetime(frame["match_date"]).dt.to_period("M")
            y = Split.labels(frame).to_numpy()
            p = np.clip(evaluation.probabilities[part], EPSILON, 1 - EPSILON)
            r = np.clip(reference.probabilities[part], EPSILON, 1 - EPSILON)
            for month in sorted(months.unique()):
                mask = (months == month).to_numpy()
                if mask.sum() < self.config.min_segment_rows:
                    continue
                model_loss = float(log_loss(y[mask], p[mask], labels=[0, 1]))
                reference_loss = float(log_loss(y[mask], r[mask], labels=[0, 1]))
                rows.append({
                    "month": str(month), "month_start": month.to_timestamp(), "period": part,
                    "matches": int(mask.sum()), "log_loss": model_loss,
                    "accuracy": float(accuracy_score(y[mask], p[mask] >= 0.5)),
                    "reference_log_loss": reference_loss, "gap_vs_reference": model_loss - reference_loss,
                })  # fmt: skip
        return pd.DataFrame(rows)

    @staticmethod
    def learning_curve(model: Pipeline, split: Split) -> pd.DataFrame | None:
        """Log loss sur l'entraînement et la validation à chaque itération (gradient boosting).

        None pour les modèles sans itérations (régression logistique, forêt aléatoire).
        """
        estimator = model[-1] if hasattr(model, "steps") else model
        if not hasattr(estimator, "staged_predict_proba"):
            return None
        preprocess = model[:-1] if hasattr(model, "steps") else None
        curves = {}
        for part in ("train", "validation"):
            features = Split.features(split.part(part))
            matrix = preprocess.transform(features) if preprocess is not None else features
            y = Split.labels(split.part(part)).to_numpy()
            curves[part] = [
                float(log_loss(y, np.clip(proba[:, 1], EPSILON, 1 - EPSILON), labels=[0, 1]))
                for proba in estimator.staged_predict_proba(matrix)
            ]
        return pd.DataFrame(
            {
                "iteration": range(1, len(curves["train"]) + 1),
                "train_log_loss": curves["train"],
                "validation_log_loss": curves["validation"],
            }
        )

    def worst_predictions(self, evaluation: Evaluation, split: Split) -> pd.DataFrame:
        """Les plus grosses surprises du test : le modèle était sûr de lui et s'est trompé."""
        frame = split.test.copy()
        p = np.clip(evaluation.probabilities["test"], EPSILON, 1 - EPSILON)
        y = Split.labels(frame).to_numpy()
        frame["p_player1"] = p
        frame["predicted_winner"] = np.where(p >= 0.5, frame["p1_name"], frame["p2_name"])
        frame["actual_winner"] = np.where(y == 1, frame["p1_name"], frame["p2_name"])
        frame["loss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        columns = ["match_date", "tourney_name", "round", "surface", "p1_name", "p1_rank", "p2_name", "p2_rank",
                   "p_player1", "predicted_winner", "actual_winner", "loss", "match_id"]  # fmt: skip
        return frame.nlargest(self.config.worst_predictions, "loss")[columns].reset_index(drop=True)


def _segment_order(segment: object) -> tuple[int, float, str]:
    """Tranches numériques dans l'ordre (0-10, 10-50, 50-200, 200+), puis les libellés, « inconnu » à la fin."""
    text = str(segment)
    head = text.split("-")[0].rstrip("+")
    if head.isdigit():
        return 0, float(head), text
    return (2 if text.startswith("inconnu") else 1), 0.0, text


# --------------------------------------------------------------------------- baselines


def elo_baseline(frame: pd.DataFrame) -> Array:
    """Baseline sans apprentissage : la probabilité Elo (globale + surface) calculée par le FeatureBuilder."""
    return frame["elo_win_prob"].to_numpy(dtype=float)


def rank_baseline(train: pd.DataFrame) -> Callable[[pd.DataFrame], Array]:
    """Baseline « classement » : régression logistique sur le seul écart de classement (log), ajustée sur le train."""
    model = LogisticRegression()
    model.fit(train[["rank_log_ratio"]].fillna(0.0), train["label"])

    def score(frame: pd.DataFrame) -> Array:
        proba: Array = model.predict_proba(frame[["rank_log_ratio"]].fillna(0.0))[:, 1]
        return proba

    return score


def _positive(model: ProbabilityModel, features: pd.DataFrame) -> Array:
    proba: Array = model.predict_proba(features)[:, 1]
    return proba
