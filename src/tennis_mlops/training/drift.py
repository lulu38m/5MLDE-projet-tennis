"""Profil des données d'entraînement et dérive par rapport à la campagne précédente.

Le profil (quantiles, valeurs manquantes, modalités) est journalisé avec chaque campagne. La suivante le relit et
calcule, feature par feature, le **PSI** (Population Stability Index) : 0 = même distribution, au-delà de 0,1 une
évolution à surveiller, au-delà de 0,2 un changement net (seuils usuels). La dérive est tracée, pas bloquante :
le circuit évolue réellement d'une saison à l'autre.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tennis_mlops.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

PSI_ALERT = 0.2
Array = npt.NDArray[np.float64]
_EPS = 1e-4


def profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Résumé de chaque feature : déciles et proportions (numériques), parts des modalités (catégorielles)."""
    numeric: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        values = frame[column].dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11))) if len(values) else np.array([])
        numeric[column] = {
            "mean": float(values.mean()) if len(values) else math.nan,
            "std": float(values.std()) if len(values) else math.nan,
            "missing_rate": float(frame[column].isna().mean()),
            "edges": edges.tolist(),
            "proportions": _proportions(values, edges).tolist(),
        }
    categorical = {
        column: {str(k): float(v) for k, v in frame[column].fillna("inconnu").value_counts(normalize=True).items()}
        for column in CATEGORICAL_FEATURES
    }
    return {"rows": len(frame), "numeric": numeric, "categorical": categorical}


def drift(previous: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    """PSI de chaque feature entre le profil précédent et `frame` ; liste des features au-delà de 0,2."""
    scores: dict[str, float] = {}
    for column, ref in previous.get("numeric", {}).items():
        if column not in frame or len(ref["edges"]) < 2:
            continue
        current = _proportions(frame[column].dropna().to_numpy(dtype=float), np.array(ref["edges"]))
        scores[column] = _psi(np.array(ref["proportions"]), current)
    for column, ref in previous.get("categorical", {}).items():
        if column not in frame:
            continue
        shares = frame[column].fillna("inconnu").value_counts(normalize=True)
        categories = sorted(set(ref) | {str(c) for c in shares.index})
        expected = np.array([ref.get(c, 0.0) for c in categories])
        actual = np.array([float(shares.get(c, 0.0)) for c in categories])
        scores[column] = _psi(expected, actual)
    over = sorted(c for c, s in scores.items() if s > PSI_ALERT)
    return {
        "psi": dict(sorted(scores.items(), key=lambda kv: -kv[1])),
        "max_psi": max(scores.values(), default=0.0),
        "features_over_alert": over,
        "alert_threshold": PSI_ALERT,
    }


def _proportions(values: Array, edges: Array) -> Array:
    if len(edges) < 2 or len(values) == 0:
        return np.array([])
    inner = edges[1:-1]
    counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(edges) - 1)
    proportions: Array = counts / counts.sum()
    return proportions


def _psi(expected: Array, actual: Array) -> float:
    if len(expected) == 0 or len(expected) != len(actual):
        return 0.0
    e, a = np.clip(expected, _EPS, None), np.clip(actual, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))
