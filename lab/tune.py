"""Réglage automatique par Optuna, sur les seules saisons `tune` : les saisons `eval` jugent ensuite le résultat.

L'objectif est la log loss moyenne des saisons de réglage : c'est une mesure continue et stable, alors que
l'accuracy change par sauts (un match de plus ou de moins) et ferait choisir au hasard entre essais proches.
L'accuracy est calculée et tracée à chaque essai ; le classement final montre les deux.

Un essai qui fait nettement moins bien que la médiane après sa première saison est arrêté (élagage).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna
import pandas as pd

from lab.evaluate import select, walk_forward

Space = Callable[[optuna.Trial], dict[str, Any]]

SPACES: dict[str, Space] = {
    "logreg": lambda t: {"C": t.suggest_float("C", 1e-3, 10, log=True)},
    "lightgbm": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": t.suggest_int("num_leaves", 4, 128, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 20, 600, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 50, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "min_split_gain": t.suggest_float("min_split_gain", 0.0, 0.5),
    },
    "xgboost": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 300, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 50, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "gamma": t.suggest_float("gamma", 0.0, 5.0),
    },
    "catboost": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth": t.suggest_int("depth", 3, 8),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 50, log=True),
        "random_strength": t.suggest_float("random_strength", 0.1, 10, log=True),
        "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 2.0),
    },
    "hgb": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 4, 64, log=True),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 20, 600, log=True),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-3, 50, log=True),
    },
}


def study(
    frame: pd.DataFrame, family: str, config: dict[str, Any], trials: int, timeout: float | None,
    on_trial: Callable[[optuna.Trial, dict[str, Any], dict[str, float]], None] | None = None,
) -> optuna.Study:  # fmt: skip
    """Cherche les meilleurs hyperparamètres, le jeu de features et le poids de récence d'une famille."""
    if family not in SPACES:
        msg = f"pas d'espace de recherche pour {family} (au choix : {', '.join(SPACES)})"
        raise ValueError(msg)
    seasons = list(config["seasons"]["tune"])
    sets = list(config["feature_sets"])

    def objective(trial: optuna.Trial) -> float:
        params = SPACES[family](trial)
        feature_set = trial.suggest_categorical("feature_set", [s for s in sets if s != "elo"])
        recency = trial.suggest_categorical("recency_half_life_years", [0, 3, 6])
        cfg = config | {"recency_half_life_years": recency}
        columns = select(frame, config["feature_sets"][feature_set])
        losses: list[float] = []
        accuracies: list[float] = []
        for step, season in enumerate(seasons):
            result = walk_forward(frame, family, params, columns, [season], cfg, keep_predictions=False)
            losses.append(result.seasons[0].log_loss)
            accuracies.append(result.seasons[0].accuracy)
            trial.report(sum(losses) / len(losses), step)
            if trial.should_prune():
                raise optuna.TrialPruned
        means = {"log_loss": sum(losses) / len(losses), "accuracy": sum(accuracies) / len(accuracies)}
        trial.set_user_attr("accuracy", means["accuracy"])
        if on_trial is not None:
            on_trial(trial, params | {"feature_set": feature_set, "recency_half_life_years": recency}, means)
        return means["log_loss"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    created = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=min(10, trials)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        study_name=f"lab-{family}",
    )  # fmt: skip
    created.optimize(objective, n_trials=trials, timeout=timeout, gc_after_trial=True)
    return created


def split_best(params: dict[str, Any]) -> tuple[dict[str, Any], str, float]:
    """Sépare les hyperparamètres du modèle, le jeu de features et la demi-vie de récence."""
    model = {k: v for k, v in params.items() if k not in {"feature_set", "recency_half_life_years"}}
    return model, str(params.get("feature_set", "all")), float(params.get("recency_half_life_years", 0))
