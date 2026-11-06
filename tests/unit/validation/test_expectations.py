"""Chaque expectation custom accepte une donnée saine et refuse exactement la ligne fautive."""

from collections.abc import Iterator
from typing import Any

import great_expectations as gx
import pandas as pd
import pytest
from great_expectations.data_context.types.base import ProgressBarsConfig
from great_expectations.expectations.expectation import Expectation

from tennis_mlops.validation.expectations import (
    ExpectBatchVolumeToMatchDraw,
    ExpectNoMatchAfterSimulatedNow,
    ExpectRankGapDistributionStable,
    ExpectTennisScoreToBeValid,
    ExpectWinnerAndLoserToDiffer,
)

Validate = Any


@pytest.fixture(scope="module")
def validate() -> Iterator[Validate]:
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    batch_definition = (
        context.data_sources.add_pandas("t").add_dataframe_asset("a").add_batch_definition_whole_dataframe("b")
    )

    def run(frame: pd.DataFrame, expectation: Expectation) -> Any:
        return batch_definition.get_batch(batch_parameters={"dataframe": frame}).validate(expectation)

    yield run


def test_tennis_score(validate: Validate) -> None:
    frame = pd.DataFrame({"score": ["6-4 6-4", "6-3 3-6 [10-8]", "6-4 2-1 RET", "W/O", None, "6-4 7-"]})
    result = validate(frame, ExpectTennisScoreToBeValid(column="score"))
    assert not result.success
    assert result.result["partial_unexpected_list"] == ["6-4 7-"]  # la valeur nulle est ignorée
    assert validate(frame.head(5), ExpectTennisScoreToBeValid(column="score")).success


def test_no_match_after_simulated_now(validate: Validate) -> None:
    frame = pd.DataFrame({"match_date": ["2019-01-07", "2019-01-13", "2019-01-14", "pas une date"]})
    result = validate(frame, ExpectNoMatchAfterSimulatedNow(column="match_date", cutoff="2019-01-14"))
    assert result.result["partial_unexpected_list"] == ["2019-01-14", "pas une date"]  # le jour même = futur
    assert validate(frame.head(2), ExpectNoMatchAfterSimulatedNow(column="match_date", cutoff="2019-01-14")).success


def test_winner_and_loser_differ(validate: Validate) -> None:
    frame = pd.DataFrame({"winner_id": [1, 2, 3], "loser_id": [2, 2, 4]})
    result = validate(frame, ExpectWinnerAndLoserToDiffer(column_A="winner_id", column_B="loser_id"))
    assert (result.success, result.result["unexpected_count"]) == (False, 1)


def test_batch_volume_matches_draw(validate: Validate) -> None:
    frame = pd.DataFrame({"x": range(7)})
    assert validate(frame, ExpectBatchVolumeToMatchDraw(value=7)).success
    result = validate(frame.head(5), ExpectBatchVolumeToMatchDraw(value=7))
    assert (result.success, result.result["observed_value"]) == (False, 5)


def test_rank_gap_distribution(validate: Validate) -> None:
    frame = pd.DataFrame({"rank_gap": [10, 40, 50, 60, 300]})
    assert validate(frame, ExpectRankGapDistributionStable(column="rank_gap", min_value=14, max_value=131)).success
    assert not validate(frame, ExpectRankGapDistributionStable(column="rank_gap", min_value=100, max_value=200)).success


def test_custom_expectations_carry_their_own_type_name() -> None:
    names = {
        ExpectTennisScoreToBeValid(column="s").expectation_type,
        ExpectWinnerAndLoserToDiffer(column_A="a", column_B="b").expectation_type,
        ExpectNoMatchAfterSimulatedNow(column="d", cutoff="2019-01-01").expectation_type,
        ExpectBatchVolumeToMatchDraw(value=1).expectation_type,
        ExpectRankGapDistributionStable(column="g", min_value=0, max_value=1).expectation_type,
    }
    assert names == {
        "expect_tennis_score_to_be_valid",
        "expect_winner_and_loser_to_differ",
        "expect_no_match_after_simulated_now",
        "expect_batch_volume_to_match_draw",
        "expect_rank_gap_distribution_stable",
    }
