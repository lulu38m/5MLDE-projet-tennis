"""Le labo : chaîne de Markov, features sans fuite du futur, modèles symétriques."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from lab.evaluate import metrics, select, walk_forward
from lab.features import FeatureEngine, feature_columns, games_share, sets_played
from lab.models import Symmetric, add_logits, make, mirror

from lab import markov
from tests.support.synthetic import synthetic_matches


def _matches(weeks: int = 40) -> pd.DataFrame:
    frame = synthetic_matches(weeks=weeks, matches_per_week=12, players=30)
    frame["match_date"] = pd.to_datetime(frame["match_date"]).dt.date
    return frame


def test_markov() -> None:
    assert markov.match(0.65, 0.65) == pytest.approx(0.5)
    assert markov.match(0.66, 0.62) > 0.5
    assert markov.match(0.66, 0.62, 5) > markov.match(0.66, 0.62, 3)  # plus long : le meilleur l'emporte plus
    assert markov.match(0.62, 0.66) == pytest.approx(1 - markov.match(0.66, 0.62), abs=1e-9)
    assert markov.game(0.5) == pytest.approx(0.5)


def test_score_parsing() -> None:
    assert games_share("6-4 7-6(5)") == pytest.approx(13 / 23)
    assert games_share("6-3 3-6 [10-8]") == pytest.approx(0.5)
    assert games_share(None) is None
    assert sets_played("6-4 3-6 7-5") == 3


def test_features_never_see_the_future() -> None:
    """Les features d'un match ne changent pas quand on retire tous les matchs qui viennent après lui."""
    matches = _matches()
    full = FeatureEngine().build(matches).set_index("match_id")
    cutoff = matches["match_date"].min() + timedelta(days=120)
    early = FeatureEngine().build(matches[matches["match_date"] <= cutoff]).set_index("match_id")
    common = full.loc[list(early.index), list(early.columns)]
    pd.testing.assert_frame_equal(common, early, check_dtype=False)


def test_a_match_does_not_see_its_own_result() -> None:
    matches = _matches(weeks=5)
    rows = FeatureEngine().build(matches)
    first = rows.iloc[0]
    assert first["a_n"] == 0 and first["b_n"] == 0 and first["elo_prob"] == pytest.approx(0.5)
    assert set(rows["label"]) == {0, 1}  # joueur 1 tiré par le hash : la cible n'est pas toujours 1


def test_symmetric_models_give_one_minus_p_when_players_are_swapped() -> None:
    frame = add_logits(FeatureEngine().build(_matches()))
    columns = select(frame, ["^d_", "_prob$", "^surface$"])
    model = make("logreg", {"C": 0.1}).fit(frame[columns], frame["label"].to_numpy())
    p, q = model.predict_proba(frame[columns]), model.predict_proba(mirror(frame[columns]))
    np.testing.assert_allclose(p + q, 1.0, atol=1e-9)
    assert isinstance(model, Symmetric)


def test_walk_forward_trains_only_on_the_past() -> None:
    frame = add_logits(FeatureEngine().build(_matches(weeks=110)))
    years = sorted(pd.to_datetime(frame["match_date"]).dt.year.unique())
    config = {"train_start": f"{years[0]}-01-01", "inner_validation_months": 6, "refit_full": True,
              "recency_half_life_years": 0}  # fmt: skip
    result = walk_forward(frame, "logreg", {"C": 0.05}, feature_columns(frame), [int(years[-1])], config)
    (season,) = result.seasons
    expected = (pd.to_datetime(frame["match_date"]).dt.year == years[-1]).sum()
    assert season.season == years[-1] and season.matches == expected
    assert 0 < season.log_loss < 1 and set(result.predictions["season"]) == {years[-1]}


def test_metrics() -> None:
    m = metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.2, 0.6, 0.4]))
    assert m["accuracy"] == 1.0 and m["auc"] == 1.0 and m["log_loss"] < 0.4


@pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost"])
def test_boostings_are_symmetric_too(family: str) -> None:
    pytest.importorskip(family)  # groupe `lab` : installé par `make lab-install`
    frame = add_logits(FeatureEngine().build(_matches(weeks=30)))
    columns = select(frame, ["^d_", "_prob$", "^surface$"])
    fast = {"lightgbm": {"n_estimators": 30}, "xgboost": {"n_estimators": 30}, "catboost": {"iterations": 30}}
    model = make(family, fast[family]).fit(frame[columns], frame["label"].to_numpy())
    p, q = model.predict_proba(frame[columns]), model.predict_proba(mirror(frame[columns]))
    np.testing.assert_allclose(p + q, 1.0, atol=1e-6)
