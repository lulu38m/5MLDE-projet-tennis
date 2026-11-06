from datetime import date

import pandas as pd
import pytest

from tennis_mlops.training.config import SplitConfig
from tennis_mlops.training.dataset import DatasetBuilder, EmptyDatasetError
from tennis_mlops.training.split import InsufficientDataError, Split, TemporalSplitter
from tests.support.synthetic import FrameSource, synthetic_matches
from tests.support.training import small_config


def test_dataset_keeps_provenance(source: FrameSource) -> None:
    dataset = DatasetBuilder(source, small_config().features).build()
    assert dataset.dataset_versions == ("synthetique",)
    assert dataset.lots["lots_valid"] == 160
    assert dataset.matches_loaded == len(source.frame)
    assert len(dataset.frame) < dataset.matches_loaded  # les forfaits ne sont pas des exemples
    assert dataset.first_date < dataset.last_date


def test_empty_database_is_an_explicit_error() -> None:
    empty = FrameSource(synthetic_matches().iloc[0:0])
    with pytest.raises(EmptyDatasetError, match="make ingest"):
        DatasetBuilder(empty, small_config().features).build()


def test_partitions_follow_each_other_without_overlap(split: Split) -> None:
    train, validation, test = (pd.to_datetime(p["match_date"]) for p in (split.train, split.validation, split.test))
    assert train.min() >= pd.Timestamp(split.train_start)
    assert train.max() < pd.Timestamp(split.validation_start) <= validation.min()
    assert validation.max() < pd.Timestamp(split.test_start) <= test.min()
    assert test.max() == pd.Timestamp(split.end)
    described = split.describe()
    assert described["split.test_rows"] == len(split.test)
    assert next(iter(Split.features(split.test).columns)) == "rank_log_ratio"


def test_six_month_windows_are_counted_back_from_the_last_match(split: Split) -> None:
    assert (pd.Timestamp(split.end) - pd.Timestamp(split.test_start)).days in range(180, 185)
    assert (pd.Timestamp(split.test_start) - pd.Timestamp(split.validation_start)).days in range(180, 185)


def test_too_small_partitions_are_refused(split: Split) -> None:
    frame = pd.concat([split.train, split.validation, split.test])
    splitter = TemporalSplitter(SplitConfig(train_start=date(2010, 6, 1), validation_months=1, test_months=1,
                                            min_rows=10_000))  # fmt: skip
    with pytest.raises(InsufficientDataError, match="trop petite"):
        splitter.split(frame)
