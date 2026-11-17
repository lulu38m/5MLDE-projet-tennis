"""Dérive des entrées : PSI des prédictions récentes contre la période d'entraînement du modèle servi."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tennis_mlops.serving.drift import DRIFT_FEATURES, DriftMonitor, magnitude, psi
from tests.support.serving import match_body, serving_world


def test_psi_is_zero_for_the_same_distribution_and_grows_with_a_shift() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    assert psi(reference, rng.normal(0, 1, 2000)) < 0.02
    assert 0.1 < psi(reference, rng.normal(0.5, 1, 2000)) < psi(reference, rng.normal(1.5, 1, 2000))
    assert psi(np.zeros(100), np.ones(10)) == 0.0  # feature constante : rien à mesurer
    assert psi(reference, np.array([])) == 0.0


def test_the_order_of_the_players_does_not_count_as_drift() -> None:
    assert magnitude("elo_diff", [-120, 120]).tolist() == [120, 120]
    assert magnitude("elo_win_prob", [0.3, 0.7]).tolist() == pytest.approx([0.2, 0.2])
    rng = np.random.default_rng(1)
    reference = magnitude("elo_diff", rng.normal(0, 100, 5000))
    flipped = magnitude("elo_diff", -np.abs(rng.normal(0, 100, 1000)))  # tous les favoris en joueur 2
    assert psi(reference, flipped) < 0.03


def test_monitor_uses_the_training_window_and_the_latest_predictions() -> None:
    days = pd.date_range("2013-01-01", "2016-12-31", freq="D")
    rng = np.random.default_rng(2)
    history = pd.DataFrame({"match_date": days.date} | {f: rng.normal(0, 1, len(days)) for f in DRIFT_FEATURES})
    monitor = DriftMonitor(window=300, reference_days=365)
    monitor.observe({f: float(v) for f, v in zip(DRIFT_FEATURES, rng.normal(0, 1, len(DRIFT_FEATURES)), strict=True)}
                    for _ in range(300))  # fmt: skip
    stable = monitor.report(history, date(2015, 12, 31))
    assert stable.sample == 300 and stable.reference_rows == 365
    assert (stable.reference_from, stable.reference_to) == (date(2014, 12, 31), date(2015, 12, 31))
    assert max(stable.psi.values()) < 0.1
    monitor.observe({"elo_diff": 4.0} for _ in range(300))  # la fenêtre ne garde que les 300 dernières
    shifted = monitor.report(history, date(2015, 12, 31))
    assert shifted.psi["elo_diff"] > 1 and set(shifted.psi) == {"elo_diff"}
    assert monitor.report(None, None).psi == {}


def test_api_exposes_drift_and_stores_the_model_age(tmp_path: Path) -> None:
    world = serving_world(tmp_path)
    data_end = (pd.to_datetime(world.frame["match_date"]).max() - timedelta(days=30)).date()
    world.registry.tags["1"]["data_window_end"] = data_end.isoformat()
    world.ready()
    for i in range(3):
        world.client.post("/predict", json=match_body(100_000 + i, 100_010 + i))
    world.flush()
    drift = world.client.get("/model").json()["drift"]
    assert drift["sample"] == 3 and set(drift["psi"]) <= set(DRIFT_FEATURES) and drift["psi"]
    assert drift["reference_to"] == data_end.isoformat()
    metrics = world.client.get("/metrics").text
    assert 'tennis_api_feature_drift_psi{feature="elo_diff"}' in metrics and "tennis_api_drift_sample 3.0" in metrics
    rows = world.rows("SELECT DISTINCT model_data_end FROM predictions")
    assert [str(r["model_data_end"]) for r in rows] == [data_end.isoformat()]
