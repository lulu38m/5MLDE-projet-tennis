"""Les modèles du labo, tous rendus **symétriques** : échanger les deux joueurs donne exactement 1 - p.

Chaque match est appris dans les deux sens (joueur 1 / joueur 2 inversés, cible inversée) et la prédiction est la
moyenne de p(A, B) et de 1 - p(B, A). Un arbre de décision n'a aucune raison d'être symétrique sans cela : sans
cette astuce, la probabilité de Djokovic contre Nadal dépendrait de l'ordre de saisie.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lab.features import CATEGORICAL

PROB_COLUMNS_SUFFIX = "_prob"
Array = npt.NDArray[Any]


class Classifier(Protocol):
    """Ce que le labo attend d'une famille de modèles."""

    def fit(self, x: pd.DataFrame, y: Array, x_val: pd.DataFrame | None, y_val: Array | None,
            weight: Array | None) -> Classifier:  # fmt: skip
        """Entraîne le modèle (validation pour l'arrêt anticipé, poids de récence)."""
        ...

    def predict_proba(self, x: pd.DataFrame) -> Array:
        """Probabilité que le joueur 1 gagne."""
        ...


def mirror(frame: pd.DataFrame) -> pd.DataFrame:
    """Le même match, joueurs inversés : a_ ↔ b_, écarts d_ opposés, probabilités 1 - p."""
    out = frame.copy()
    for col in frame.columns:
        if col.startswith("a_") and f"b_{col[2:]}" in frame:
            out[col], out[f"b_{col[2:]}"] = frame[f"b_{col[2:]}"], frame[col]
        elif col.startswith("d_"):
            out[col] = -frame[col]
        elif col.endswith(PROB_COLUMNS_SUFFIX):
            out[col] = 1 - frame[col]
        elif col.endswith("_logit"):
            out[col] = -frame[col]
    return out


def add_logits(frame: pd.DataFrame) -> pd.DataFrame:
    """log(p / (1 - p)) de chaque probabilité : plus facile à combiner linéairement."""
    out = frame.copy()
    for col in [c for c in frame.columns if c.endswith(PROB_COLUMNS_SUFFIX)]:
        p = frame[col].clip(1e-4, 1 - 1e-4)
        out[col.removesuffix(PROB_COLUMNS_SUFFIX) + "_logit"] = np.log(p / (1 - p))
    return out


@dataclass
class Symmetric:
    """Apprend sur les deux sens de chaque match, prédit la moyenne des deux sens."""

    inner: Classifier

    def fit(self, x: pd.DataFrame, y: Array, x_val: pd.DataFrame | None = None, y_val: Array | None = None,
            weight: Array | None = None) -> Symmetric:  # fmt: skip
        """Apprend sur le match et sur son miroir."""
        xx = pd.concat([x, mirror(x)], ignore_index=True)
        yy = np.concatenate([y, 1 - y])
        ww = None if weight is None else np.concatenate([weight, weight])
        if x_val is not None and y_val is not None:
            x_val, y_val = pd.concat([x_val, mirror(x_val)], ignore_index=True), np.concatenate([y_val, 1 - y_val])
        self.inner.fit(xx, yy, x_val, y_val, ww)
        return self

    def predict_proba(self, x: pd.DataFrame) -> Array:
        """Moyenne de p(A, B) et de 1 - p(B, A)."""
        return 0.5 * (self.inner.predict_proba(x) + 1 - self.inner.predict_proba(mirror(x)))


# ---------------------------------------------------------------------- familles


CATEGORIES = {
    "surface": ["Hard", "Clay", "Grass", "Carpet", "inconnu"],
    "tourney_level": ["G", "M", "A", "F", "D", "O", "inconnu"],
    "round": ["R128", "R64", "R32", "R16", "QF", "SF", "F", "RR", "BR", "ER", "inconnu"],
}


def _categorical(x: pd.DataFrame) -> pd.DataFrame:
    """Mêmes catégories à l'entraînement et à la prédiction (une valeur jamais vue devient manquante)."""
    out = x.copy()
    for col in CATEGORICAL:
        if col in out:
            out[col] = pd.Categorical(out[col], categories=CATEGORIES.get(col))
    return out


class LogReg:
    """Régression logistique sur les features standardisées (+ catégories en one-hot)."""

    def __init__(self, C: float = 0.05, **_: object) -> None:  # noqa: N803
        self.C = C
        self.model: Pipeline | None = None

    def fit(self, x: pd.DataFrame, y: Array, x_val: Any = None, y_val: Any = None, weight: Any = None) -> LogReg:  # noqa: ANN401
        """Entraîne le modèle."""
        numeric = [c for c in x.columns if c not in CATEGORICAL]
        cats = [c for c in CATEGORICAL if c in x]
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cats),
        ])  # fmt: skip
        self.model = Pipeline([("pre", pre), ("lr", LogisticRegression(C=self.C, max_iter=3000))])
        self.model.fit(x, y, lr__sample_weight=weight)
        return self

    def predict_proba(self, x: pd.DataFrame) -> Array:
        """Probabilité que le joueur 1 gagne."""
        assert self.model is not None
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)


class Lgbm:
    """LightGBM, arrêt anticipé sur la saison de validation interne."""

    iterations_key = "n_estimators"

    def best_n(self) -> int | None:
        """Nombre d'arbres retenu par l'arrêt anticipé."""
        return int(self.model.best_iteration_) if getattr(self.model, "best_iteration_", None) else None

    def __init__(self, **params: Any) -> None:  # noqa: ANN401
        self.params = {"n_estimators": 3000, "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 50,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 1.0,
                       "verbose": -1, "random_state": 42} | params  # fmt: skip
        self.model: Any = None

    def fit(self, x: pd.DataFrame, y: Array, x_val: pd.DataFrame | None, y_val: Array | None,
            weight: Array | None) -> Lgbm:  # fmt: skip
        """Entraîne le modèle."""
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(**self.params)
        kwargs: dict[str, Any] = {"sample_weight": weight}
        if x_val is not None:
            stop = lgb.early_stopping(100, verbose=False)
            kwargs |= {"eval_set": [(_categorical(x_val), y_val)], "callbacks": [stop]}
        self.model.fit(_categorical(x), y, **kwargs)
        return self

    def predict_proba(self, x: pd.DataFrame) -> Array:
        """Probabilité que le joueur 1 gagne."""
        return np.asarray(self.model.predict_proba(_categorical(x))[:, 1], dtype=float)


class Xgb:
    """XGBoost (histogrammes, catégories natives)."""

    iterations_key = "n_estimators"

    def best_n(self) -> int | None:
        """Nombre d'arbres retenu par l'arrêt anticipé."""
        best = getattr(self.model, "best_iteration", None)
        return int(best) + 1 if best is not None else None

    def __init__(self, **params: Any) -> None:  # noqa: ANN401
        self.params = {"n_estimators": 3000, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 20,
                       "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 1.0, "tree_method": "hist",
                       "enable_categorical": True, "eval_metric": "logloss", "early_stopping_rounds": 100,
                       "random_state": 42} | params  # fmt: skip
        self.model: Any = None

    def fit(self, x: pd.DataFrame, y: Array, x_val: pd.DataFrame | None, y_val: Array | None,
            weight: Array | None) -> Xgb:  # fmt: skip
        """Entraîne le modèle."""
        import xgboost as xgb

        params = dict(self.params)
        if x_val is None:
            params.pop("early_stopping_rounds", None)
        self.model = xgb.XGBClassifier(**params)
        evals = [(_categorical(x_val), y_val)] if x_val is not None else None
        self.model.fit(_categorical(x), y, sample_weight=weight, eval_set=evals, verbose=False)
        return self

    def predict_proba(self, x: pd.DataFrame) -> Array:
        """Probabilité que le joueur 1 gagne."""
        return np.asarray(self.model.predict_proba(_categorical(x))[:, 1], dtype=float)


class CatBoost:
    """CatBoost (boosting ordonné, robuste au sur-apprentissage)."""

    iterations_key = "iterations"

    def best_n(self) -> int | None:
        """Nombre d'arbres retenu par l'arrêt anticipé."""
        best = self.model.get_best_iteration()
        return int(best) + 1 if best is not None else None

    def __init__(self, **params: Any) -> None:  # noqa: ANN401
        self.params = {"iterations": 3000, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0,
                       "loss_function": "Logloss", "random_seed": 42, "verbose": False,
                       "allow_writing_files": False, "thread_count": -1} | params  # fmt: skip
        self.model: Any = None

    def fit(self, x: pd.DataFrame, y: Array, x_val: pd.DataFrame | None, y_val: Array | None,
            weight: Array | None) -> CatBoost:  # fmt: skip
        """Entraîne le modèle."""
        from catboost import CatBoostClassifier

        cats = [c for c in CATEGORICAL if c in x]
        self.model = CatBoostClassifier(**self.params, cat_features=cats)
        kwargs: dict[str, Any] = {"sample_weight": weight}
        if x_val is not None:
            kwargs |= {"eval_set": (x_val, y_val), "early_stopping_rounds": 150}
        self.model.fit(x, y, **kwargs)
        return self

    def predict_proba(self, x: pd.DataFrame) -> Array:
        """Probabilité que le joueur 1 gagne."""
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)


class Hgb:
    """HistGradientBoosting de scikit-learn (la famille déjà présente dans le pipeline)."""

    def __init__(self, **params: Any) -> None:  # noqa: ANN401
        self.params = {"learning_rate": 0.05, "max_iter": 2000, "max_leaf_nodes": 31, "min_samples_leaf": 50,
                       "l2_regularization": 1.0, "early_stopping": True, "validation_fraction": 0.1,
                       "n_iter_no_change": 50, "random_state": 42,
                       "categorical_features": "from_dtype"} | params  # fmt: skip
        self.model: Any = None

    def fit(self, x: pd.DataFrame, y: Array, x_val: Any = None, y_val: Any = None, weight: Any = None) -> Hgb:  # noqa: ANN401
        """Entraîne le modèle."""
        self.model = HistGradientBoostingClassifier(**self.params).fit(_categorical(x), y, sample_weight=weight)
        return self

    def predict_proba(self, x: pd.DataFrame) -> Array:
        """Probabilité que le joueur 1 gagne."""
        return np.asarray(self.model.predict_proba(_categorical(x))[:, 1], dtype=float)


class EloOnly:
    """Référence : la seule probabilité Elo mélangée, recalibrée par une régression logistique à une variable."""

    def __init__(self, column: str = "blend_elo_logit", **_: Any) -> None:  # noqa: ANN401
        self.column = column
        self.model = LogisticRegression(C=1e6, max_iter=1000)

    def fit(self, x: pd.DataFrame, y: Array, x_val: Any = None, y_val: Any = None, weight: Any = None) -> EloOnly:  # noqa: ANN401
        """Entraîne le modèle."""
        self.model.fit(x[[self.column]], y, sample_weight=weight)
        return self

    def predict_proba(self, x: pd.DataFrame) -> Array:
        """Probabilité que le joueur 1 gagne."""
        return np.asarray(self.model.predict_proba(x[[self.column]])[:, 1], dtype=float)


FAMILIES: dict[str, Callable[..., Classifier]] = {
    "elo": EloOnly, "logreg": LogReg, "hgb": Hgb, "lightgbm": Lgbm, "xgboost": Xgb, "catboost": CatBoost,
}  # fmt: skip


def make(family: str, params: dict[str, Any] | None = None) -> Symmetric:
    """Un modèle symétrique de la famille demandée."""
    if family not in FAMILIES:
        msg = f"famille inconnue : {family} (au choix : {', '.join(FAMILIES)})"
        raise ValueError(msg)
    return Symmetric(FAMILIES[family](**(params or {})))


def logit(p: Array) -> Array:
    """log(p / (1 - p))."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(p / (1 - p)))


def sigmoid(z: Array) -> Array:
    """1 / (1 + e^-z)."""
    return np.asarray(1 / (1 + np.exp(-z)))
