"""La référence à battre : les features et le modèle du pipeline actuel, évalués sur les mêmes saisons."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import yaml

from lab.data import load_matches
from lab.evaluate import RunResult, SeasonResult, metrics
from tennis_mlops.features.builder import FEATURE_COLUMNS, FeatureBuilder, FeatureConfig
from tennis_mlops.training.config import DEFAULT_TRAIN_CONFIG_PATH, ModelKind
from tennis_mlops.training.pipeline import CandidateSpec, PipelineFactory


def pipeline_reference(seasons: list[int], config: dict[str, Any], C: float = 0.05) -> RunResult:  # noqa: N803
    """Régression logistique C = 0,05 sur les features du pipeline (la version servie aujourd'hui)."""
    train_config = yaml.safe_load(DEFAULT_TRAIN_CONFIG_PATH.read_text(encoding="utf-8"))
    frame = FeatureBuilder(FeatureConfig(**train_config["features"])).build(load_matches())
    dates = pd.to_datetime(frame["match_date"])
    spec = CandidateSpec(ModelKind.LOGISTIC_REGRESSION, {"C": C})
    result = RunResult("pipeline", {"C": C})
    for season in seasons:
        began = time.perf_counter()
        train = frame[(dates >= pd.Timestamp(config["train_start"])) & (dates < pd.Timestamp(season, 1, 1))]
        test = frame[dates.dt.year == season]
        model = PipelineFactory().create(spec).fit(train[list(FEATURE_COLUMNS)], train["label"])
        p = model.predict_proba(test[list(FEATURE_COLUMNS)])[:, 1]
        m = metrics(test["label"].to_numpy(), p)
        result.seasons.append(SeasonResult.of(season, len(test), m, time.perf_counter() - began))
    return result
