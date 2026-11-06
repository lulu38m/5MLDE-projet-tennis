"""Contenu des suites : ce qui bloque un lot, et ce qui est seulement surveillé.

- **Suite bloquante** (`atp_lot_blocking`) : contrat de schéma, domaines de valeurs, règles métier. Un seul échec
  = lot en quarantaine, aucun stockage, entraînement non déclenché.
- **Suite de surveillance** (`atp_lot_monitoring`) : dérive de distribution. Un écart est tracé et signalé, mais
  ne bloque pas : un tournoi atypique (une Coupe Davis, un tableau de qualifiés) n'est pas une donnée fausse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from great_expectations import expectations as gxe
from great_expectations.expectations.expectation import Expectation

from tennis_mlops.validation.expectations import (
    ExpectBatchVolumeToMatchDraw,
    ExpectNoMatchAfterSimulatedNow,
    ExpectRankGapDistributionStable,
    ExpectTennisScoreToBeValid,
    ExpectWinnerAndLoserToDiffer,
)

BLOCKING_SUITE = "atp_lot_blocking"
MONITORING_SUITE = "atp_lot_monitoring"
SUITES_VERSION = "1.0"

REQUIRED_COLUMNS = (
    "match_id",
    "tourney_id",
    "tourney_level",
    "surface",
    "round",
    "best_of",
    "match_date",
    "status",
    "score",
    "minutes",
    "winner_id",
    "loser_id",
    "winner_rank",
    "loser_rank",
)
LEVELS = ["G", "M", "A", "D", "F", "O"]
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "BR", "RR"]
STATUSES = ["completed", "retired", "walkover", "defaulted", "abandoned"]
INTEGER_TYPES = ["int", "int32", "int64", "Int64"]
NUMBER_TYPES = [*INTEGER_TYPES, "float", "float64"]
ISO_DATE = r"^\d{4}-\d{2}-\d{2}$"
MAX_MINUTES = 1200  # le plus long match du dataset dure 1 146 min (donnée source), Isner-Mahut 665
MAX_RANK = 5000


@dataclass(frozen=True, slots=True)
class LotExpectations:
    """Paramètres propres à un lot."""

    simulated_now: date
    expected_rows: int | None = None


@dataclass(frozen=True, slots=True)
class RankGapReference:
    """Plage historique de la médiane de l'écart de classement, pour un niveau de tournoi."""

    level: str
    low: float
    high: float
    tournaments: int


def blocking_expectations(lot: LotExpectations) -> list[Expectation]:
    """Règles bloquantes : schéma, valeurs, métier."""
    rules: list[Expectation] = [gxe.ExpectColumnToExist(column=c) for c in REQUIRED_COLUMNS]
    rules += [
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
        *(gxe.ExpectColumnValuesToNotBeNull(column=c) for c in ("match_id", "match_date", "winner_id", "loser_id")),
        gxe.ExpectColumnValuesToBeUnique(column="match_id"),
        gxe.ExpectColumnValuesToBeInTypeList(column="winner_id", type_list=INTEGER_TYPES),
        gxe.ExpectColumnValuesToBeInTypeList(column="loser_id", type_list=INTEGER_TYPES),
        gxe.ExpectColumnValuesToBeInTypeList(column="minutes", type_list=NUMBER_TYPES),
        gxe.ExpectColumnValuesToBeInSet(column="tourney_level", value_set=LEVELS),
        gxe.ExpectColumnValuesToBeInSet(column="surface", value_set=SURFACES),
        gxe.ExpectColumnValuesToBeInSet(column="round", value_set=ROUNDS),
        gxe.ExpectColumnValuesToBeInSet(column="best_of", value_set=[3, 5]),
        gxe.ExpectColumnValuesToBeInSet(column="status", value_set=STATUSES),
        gxe.ExpectColumnValuesToMatchRegex(column="match_date", regex=ISO_DATE),
        gxe.ExpectColumnValuesToBeBetween(column="minutes", min_value=0, max_value=MAX_MINUTES),
        gxe.ExpectColumnValuesToBeBetween(column="winner_rank", min_value=1, max_value=MAX_RANK),
        gxe.ExpectColumnValuesToBeBetween(column="loser_rank", min_value=1, max_value=MAX_RANK),
        ExpectTennisScoreToBeValid(column="score"),
        ExpectWinnerAndLoserToDiffer(column_A="winner_id", column_B="loser_id"),
        ExpectNoMatchAfterSimulatedNow(column="match_date", cutoff=lot.simulated_now.isoformat()),
    ]
    if lot.expected_rows is not None:
        rules.append(ExpectBatchVolumeToMatchDraw(value=lot.expected_rows))
    return rules


def monitoring_expectations(reference: RankGapReference | None) -> list[Expectation]:
    """Règles de surveillance (non bloquantes) ; vide tant que l'historique est insuffisant."""
    if reference is None:
        return []
    return [ExpectRankGapDistributionStable(column="rank_gap", min_value=reference.low, max_value=reference.high)]
