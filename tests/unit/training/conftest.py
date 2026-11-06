import pytest

from tennis_mlops.training.dataset import DatasetBuilder
from tennis_mlops.training.split import Split, TemporalSplitter
from tests.support.synthetic import FrameSource, synthetic_matches
from tests.support.training import small_config


@pytest.fixture(scope="session")
def source() -> FrameSource:
    return FrameSource(synthetic_matches())


@pytest.fixture(scope="session")
def split(source: FrameSource) -> Split:
    config = small_config()
    dataset = DatasetBuilder(source, config.features).build()
    return TemporalSplitter(config.split).split(dataset.frame)
