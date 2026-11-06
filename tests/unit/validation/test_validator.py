import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mock_source.chaos import break_schema
from tennis_mlops.ingestion.client import MockSourceClient
from tennis_mlops.ingestion.lots import Lot, results_frame, tournament_lot
from tennis_mlops.validation.suites import RankGapReference
from tennis_mlops.validation.validator import LotValidator
from tests.support.mock_world import MockEnv


@pytest.fixture(scope="module")
def validator(tmp_path_factory: pytest.TempPathFactory) -> LotValidator:
    return LotValidator(tmp_path_factory.mktemp("gx"))


@pytest.fixture
def clean_lot(env: MockEnv) -> Lot:
    env.at(date(2019, 1, 14))
    return tournament_lot(MockSourceClient("", 5, http=env.client).tournament_results("2019-0001"))


def with_frame(lot: Lot, frame: Any) -> Lot:
    return replace(lot, frame=frame)


def failed(validator: LotValidator, lot: Lot) -> set[str]:
    report = validator.validate(lot)
    assert not report.success
    return {f"{c.expectation}:{c.target}" for c in report.failures}


def test_clean_lot_passes_and_produces_data_docs(validator: LotValidator, clean_lot: Lot) -> None:
    report = validator.validate(clean_lot)
    assert report.success, [c.describe() for c in report.failures]
    assert report.failures == [] and report.warnings == []
    assert report.docs_index is not None and Path(report.docs_index).is_file()
    assert json.loads(report.to_json())["lot_id"] == "2019-0001"


def test_each_business_rule_blocks_its_defect(validator: LotValidator, clean_lot: Lot) -> None:
    frame = clean_lot.frame.copy()
    frame.loc[0, "loser_id"] = frame.loc[0, "winner_id"]
    frame.loc[1, "score"] = "6-4 7-"
    frame.loc[2, "match_date"] = "2019-02-20"
    frame.loc[3, "surface"] = "Moon"
    frame.loc[4, "minutes"] = -12
    assert failed(validator, with_frame(clean_lot, frame)) == {
        "expect_winner_and_loser_to_differ:winner_id/loser_id",
        "expect_tennis_score_to_be_valid:score",
        "expect_no_match_after_simulated_now:match_date",
        "expect_column_values_to_be_in_set:surface",
        "expect_column_values_to_be_between:minutes",
    }


def test_missing_rows_break_the_volume_rule(validator: LotValidator, clean_lot: Lot) -> None:
    assert failed(validator, with_frame(clean_lot, clean_lot.frame.head(5))) == {
        "expect_batch_volume_to_match_draw:table"
    }


def test_broken_schema_is_blocked(validator: LotValidator, clean_lot: Lot) -> None:
    broken = results_frame(break_schema(clean_lot.payload.data)["results"])
    failures = failed(validator, with_frame(clean_lot, broken))
    assert {
        "expect_column_to_exist:match_id",
        "expect_column_to_exist:score",
        "expect_column_to_exist:winner_id",
    } <= failures
    assert "expect_column_values_to_be_in_type_list:minutes" in failures


def test_drift_is_reported_without_blocking(validator: LotValidator, clean_lot: Lot) -> None:
    report = validator.validate(clean_lot, RankGapReference(level="A", low=500, high=900, tournaments=40))
    assert report.success
    assert [w.expectation for w in report.warnings] == ["expect_rank_gap_distribution_stable"]


def test_a_check_that_cannot_run_blocks_the_lot(validator: LotValidator, clean_lot: Lot) -> None:
    report = validator.validate(with_frame(clean_lot, "pas un DataFrame"))
    assert not report.success
    assert report.failures[0].error is not None
