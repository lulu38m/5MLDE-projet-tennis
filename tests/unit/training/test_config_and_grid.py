from datetime import date

import pytest
from pydantic import ValidationError

from tennis_mlops.training.config import DEFAULT_TRAIN_CONFIG_PATH, CandidateGrid, ModelKind, SplitConfig, TrainConfig
from tennis_mlops.training.pipeline import CandidateSpec, expand_grid
from tests.support.training import small_config


def test_project_train_config_is_valid() -> None:
    config = TrainConfig.from_yaml(DEFAULT_TRAIN_CONFIG_PATH)
    specs = expand_grid(config)
    assert len(specs) == 8
    assert len({s.name for s in specs}) == len(specs)  # noms de runs uniques
    assert config.selection_metric == "log_loss"


def test_grid_is_the_cartesian_product_in_file_order() -> None:
    specs = expand_grid(small_config())
    assert [s.name for s in specs] == ["logreg C=0.1", "logreg C=1.0", "hgb depth=3 iter=30"]
    assert specs[2] == CandidateSpec(ModelKind.HIST_GRADIENT_BOOSTING, {"max_depth": 3, "max_iter": 30}, 42)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"candidates": (CandidateGrid(model=ModelKind.RANDOM_FOREST), CandidateGrid(model=ModelKind.RANDOM_FOREST))},
         "qu'une fois"),
        ({"selection_metric": "accuracy"}, "selection_metric"),
        ({"evaluation": {"rank_gap_edges": (50, 10)}}, "croissantes"),
    ],
)  # fmt: skip
def test_invalid_configs_are_refused(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        small_config(**override)


def test_empty_grid_values_are_refused() -> None:
    with pytest.raises(ValidationError, match="sans valeur"):
        CandidateGrid(model=ModelKind.LOGISTIC_REGRESSION, grid={"C": ()})


def test_unknown_keys_are_refused(tmp_path: object) -> None:
    with pytest.raises(ValidationError):
        SplitConfig.model_validate({"train_start": date(2011, 1, 1), "validation_months": 6, "test_months": 6,
                                    "tst_months": 3})  # fmt: skip


def test_yaml_root_must_be_a_mapping(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "train.yaml"  # type: ignore[operator]
    path.write_text("- liste\n", encoding="utf-8")
    with pytest.raises(TypeError, match="objet YAML"):
        TrainConfig.from_yaml(path)
