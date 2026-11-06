"""Les 5 expectations custom du projet (Great Expectations 1.x).

Trois s'appuient sur une **métrique custom** (une règle métier calculée ligne par ligne) :

| Expectation | Métrique | Ce qu'elle bloque |
|---|---|---|
| `ExpectTennisScoreToBeValid` | `column_values.valid_tennis_score` | score illisible (`6-4 7-`) |
| `ExpectWinnerAndLoserToDiffer` | `column_pair_values.different` | vainqueur = perdant |
| `ExpectNoMatchAfterSimulatedNow` | `column_values.before_date` | match daté après l'horloge (fuite du futur) |

Deux **spécialisent** une expectation existante avec la règle du projet (méthode recommandée par GX 1.x) :

| Expectation | Base | Ce qu'elle vérifie |
|---|---|---|
| `ExpectBatchVolumeToMatchDraw` | `ExpectTableRowCountToEqual` | nb de matchs du lot = `expected_matches` |
| `ExpectRankGapDistributionStable` | `ExpectColumnMedianToBeBetween` | écart de classement médian (surveillance) |

Les métriques s'enregistrent auprès de GX à l'import de ce module : il doit être importé avant de valider.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.expectations import ExpectColumnMedianToBeBetween, ExpectTableRowCountToEqual
from great_expectations.expectations.expectation import ColumnMapExpectation, ColumnPairMapExpectation
from great_expectations.expectations.metrics.map_metric_provider import (
    ColumnMapMetricProvider,
    ColumnPairMapMetricProvider,
    column_condition_partial,
    column_pair_condition_partial,
)

from tennis_mlops.data.score import MatchStatus, classify_score

# --------------------------------------------------------------------------- métriques custom


class ColumnValuesValidTennisScore(ColumnMapMetricProvider):
    """Vrai si le score se lit comme un score de tennis (sets, tie-breaks, RET, W/O, DEF…)."""

    condition_metric_name = "column_values.valid_tennis_score"

    @column_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column: pd.Series[Any], **kwargs: object) -> pd.Series[Any]:  # noqa: N805
        return column.map(lambda score: classify_score(str(score)) is not MatchStatus.UNKNOWN)


class ColumnValuesBeforeDate(ColumnMapMetricProvider):
    """Vrai si la date (ISO) est strictement antérieure à `cutoff` ; une date illisible est fausse."""

    condition_metric_name = "column_values.before_date"
    condition_value_keys = ("cutoff",)

    @column_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column: pd.Series[Any], cutoff: str, **kwargs: object) -> pd.Series[Any]:  # noqa: N805
        return pd.to_datetime(column, errors="coerce", format="%Y-%m-%d") < pd.Timestamp(cutoff)


class ColumnPairValuesDifferent(ColumnPairMapMetricProvider):
    """Vrai si les deux colonnes ont des valeurs différentes sur la ligne."""

    condition_metric_name = "column_pair_values.different"
    condition_domain_keys = (
        "batch_id",
        "table",
        "column_A",
        "column_B",
        "row_condition",
        "condition_parser",
        "ignore_row_if",
    )
    condition_value_keys = ()

    @column_pair_condition_partial(engine=PandasExecutionEngine)  # type: ignore[untyped-decorator]
    def _pandas(cls, column_A: pd.Series[Any], column_B: pd.Series[Any], **kwargs: object) -> pd.Series[Any]:  # noqa: N803, N805
        return column_A != column_B


# --------------------------------------------------------------------------- expectations sur métriques custom


class ExpectTennisScoreToBeValid(ColumnMapExpectation):
    """Chaque score est un score de tennis lisible (les valeurs nulles sont ignorées)."""

    map_metric = "column_values.valid_tennis_score"
    success_keys = ("mostly",)


class ExpectNoMatchAfterSimulatedNow(ColumnMapExpectation):
    """Aucun match n'est daté le jour simulé ou après : la source ne doit jamais révéler le futur."""

    cutoff: str
    map_metric = "column_values.before_date"
    success_keys = ("cutoff", "mostly")


class ExpectWinnerAndLoserToDiffer(ColumnPairMapExpectation):
    """Sur chaque ligne, les deux colonnes (vainqueur, perdant) désignent deux joueurs différents."""

    ignore_row_if: Literal["both_values_are_missing", "either_value_is_missing", "neither"] = "both_values_are_missing"
    map_metric = "column_pair_values.different"
    success_keys = ("column_A", "column_B", "ignore_row_if", "mostly")
    args_keys = ("column_A", "column_B")


# --------------------------------------------------------------------------- expectations spécialisées


class ExpectBatchVolumeToMatchDraw(ExpectTableRowCountToEqual):
    """Le lot contient exactement le nombre de matchs annoncé par la source (`expected_matches`)."""

    description: str | None = "Volume du lot = nombre de matchs du tableau (`expected_matches`)"


class ExpectRankGapDistributionStable(ExpectColumnMedianToBeBetween):
    """La médiane de l'écart de classement reste dans la plage observée sur l'historique du même niveau."""

    description: str | None = "Écart de classement médian dans la plage historique du même niveau"
