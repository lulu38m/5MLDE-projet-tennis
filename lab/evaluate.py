"""Validation « walk-forward » : chaque saison est prédite par un modèle entraîné sur les seules saisons d'avant.

Trois groupes de saisons (`lab/config.yaml`) : **tune** (Optuna ne regarde que celles-ci), **eval** (le classement
des essais), **holdout** (2026, le test final, seulement sur demande). Les métriques : log loss et accuracy à
égalité, plus le Brier et l'AUC.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from lab import models
from lab.data import RAW_DIR, load_matches
from lab.features import CATEGORICAL, FeatureEngine, FeatureParams, feature_columns

CONFIG = Path(__file__).with_name("config.yaml")
STATE = Path("state/lab")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    """La configuration du labo."""
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------- jeu de features (mis en cache)


def feature_frame(config: dict[str, Any], refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Les features de tous les matchs, recalculées seulement si les données ou les réglages ont changé."""
    params = FeatureParams(**config["features"])
    manifest = (RAW_DIR / "manifest.json").read_bytes() if (RAW_DIR / "manifest.json").exists() else b""
    source = Path(__file__).with_name("features.py").read_bytes() + Path(__file__).with_name("markov.py").read_bytes()
    key = hashlib.sha256(manifest + json.dumps(asdict(params), sort_keys=True).encode() + source).hexdigest()[:12]
    path = STATE / f"features-{key}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path), key
    started = time.perf_counter()
    frame = FeatureEngine(params).build(load_matches())
    frame = models.add_logits(frame)
    STATE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    seconds = time.perf_counter() - started
    print(f"features : {len(frame)} matchs, {len(feature_columns(frame))} colonnes, {seconds:.0f} s")
    return frame, key


def select(frame: pd.DataFrame, patterns: list[str]) -> list[str]:
    """Colonnes de features (numériques et catégories) qui correspondent à l'une des expressions."""
    candidates = [*feature_columns(frame), *CATEGORICAL]
    return [c for c in candidates if any(re.search(p, c) for p in patterns)]


# ---------------------------------------------------------------------- évaluation


@dataclass
class SeasonResult:
    """Une saison prédite."""

    season: int
    matches: int
    log_loss: float
    accuracy: float
    brier: float
    auc: float
    seconds: float
    trees: int | None = None

    @classmethod
    def of(cls, season: int, matches: int, scores: dict[str, float], seconds: float,
           trees: int | None = None) -> SeasonResult:  # fmt: skip
        """À partir du dictionnaire de `metrics`."""
        return cls(season, matches, scores["log_loss"], scores["accuracy"], scores["brier"], scores["auc"], seconds,
                   trees)  # fmt: skip


@dataclass
class RunResult:
    """Les saisons d'un essai, et leurs prédictions (pour les ensembles)."""

    family: str
    params: dict[str, Any]
    seasons: list[SeasonResult] = field(default_factory=list)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance: pd.Series[float] | None = None

    def mean(self, group: list[int]) -> dict[str, float]:
        """Moyenne des saisons d'un groupe (log loss, accuracy, Brier, AUC)."""
        rows = [s for s in self.seasons if s.season in group]
        if not rows:
            return {}
        return {m: float(np.mean([getattr(s, m) for s in rows])) for m in ("log_loss", "accuracy", "brier", "auc")}


def metrics(y: npt.NDArray[Any], p: npt.NDArray[Any]) -> dict[str, float]:
    """Log loss, accuracy, Brier, AUC."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": float(log_loss(y, p, labels=[0, 1])), "accuracy": float(accuracy_score(y, p > 0.5)),
            "brier": float(brier_score_loss(y, p)), "auc": float(roc_auc_score(y, p))}  # fmt: skip


def season_of(frame: pd.DataFrame) -> pd.Series[int]:
    """L'année de chaque match."""
    return pd.to_datetime(frame["match_date"]).dt.year


def weights(train: pd.DataFrame, half_life_years: float) -> npt.NDArray[Any] | None:
    """Poids de récence : 1 pour la dernière saison, divisé par deux tous les `half_life_years`."""
    if not half_life_years:
        return None
    age = (pd.to_datetime(train["match_date"]).max() - pd.to_datetime(train["match_date"])).dt.days / 365.25
    return np.asarray(0.5 ** (age / half_life_years))


def walk_forward(
    frame: pd.DataFrame, family: str, params: dict[str, Any], columns: list[str], seasons: list[int],
    config: dict[str, Any], keep_predictions: bool = True,
) -> RunResult:  # fmt: skip
    """Entraîne et prédit chaque saison demandée, sans jamais utiliser une saison postérieure."""
    result = RunResult(family, params)
    years = season_of(frame)
    start = pd.Timestamp(config["train_start"])
    dates = pd.to_datetime(frame["match_date"])
    preds = []
    for season in seasons:
        began = time.perf_counter()
        cutoff = pd.Timestamp(date(season, 1, 1))
        train, test = frame[(dates >= start) & (dates < cutoff)], frame[years == season]
        if test.empty:
            continue
        model, trees = _fit(family, params, train, columns, config)
        p = model.predict_proba(test[columns])
        m = metrics(test["label"].to_numpy(), p)
        result.seasons.append(SeasonResult.of(season, len(test), m, time.perf_counter() - began, trees))
        if keep_predictions:
            preds.append(pd.DataFrame({"match_id": test["match_id"].to_numpy(), "season": season,
                                       "label": test["label"].to_numpy(), "p": p}))  # fmt: skip
        result.importance = _importance(model, columns)
    result.predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return result


def _fit(family: str, params: dict[str, Any], train: pd.DataFrame, columns: list[str],
         config: dict[str, Any]) -> tuple[models.Symmetric, int | None]:  # fmt: skip
    """Arrêt anticipé sur les derniers mois du jeu d'entraînement, puis (option) réentraînement sur tout le jeu."""
    dates = pd.to_datetime(train["match_date"])
    split = dates.max() - pd.DateOffset(months=int(config["inner_validation_months"]))
    fit, val = train[dates <= split], train[dates > split]
    half_life = float(config.get("recency_half_life_years") or 0)
    model = models.make(family, params)
    inner = model.inner
    if not isinstance(inner, models.Lgbm | models.Xgb | models.CatBoost):
        model.fit(train[columns], train["label"].to_numpy(), weight=weights(train, half_life))
        return model, None
    model.fit(fit[columns], fit["label"].to_numpy(), val[columns], val["label"].to_numpy(), weights(fit, half_life))
    trees = inner.best_n()
    if not config.get("refit_full") or not trees:
        return model, trees
    final = models.make(family, params | {inner.iterations_key: max(int(trees * 1.1), 10)})
    final.fit(train[columns], train["label"].to_numpy(), weight=weights(train, half_life))
    return final, trees


def _importance(model: models.Symmetric, columns: list[str]) -> pd.Series[float] | None:
    inner: Any = model.inner.model if hasattr(model.inner, "model") else None
    values = getattr(inner, "feature_importances_", None)
    if values is None and inner is not None and hasattr(inner, "get_feature_importance"):
        values = inner.get_feature_importance()
    if values is None or len(values) != len(columns):
        return None
    importance: pd.Series[float] = pd.Series(np.asarray(values, dtype=float), index=columns)
    return importance.sort_values(ascending=False)
