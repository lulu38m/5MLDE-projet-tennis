"""Jeu d'entraînement : les matchs **validés** (lots passés par Great Expectations) transformés en features."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import pandas as pd

from tennis_mlops.features import FEATURES_VERSION, FeatureBuilder, FeatureConfig


class MatchSource(Protocol):
    """D'où viennent les matchs (PostgreSQL en vrai, un DataFrame dans les tests)."""

    def matches_frame(self) -> pd.DataFrame:
        """Tous les matchs stockés, schéma de la table `matches`."""
        ...

    def lots_summary(self) -> dict[str, Any]:
        """Bilan des lots ingérés (valides, en quarantaine, dernier lot)."""
        ...


@dataclass(frozen=True)
class Dataset:
    """Features de tous les matchs joués, dans l'ordre chronologique, et leur provenance."""

    frame: pd.DataFrame
    matches_loaded: int
    dataset_versions: tuple[str, ...]
    lots: dict[str, Any] = field(default_factory=dict)
    build_seconds: float = 0.0
    features_version: str = FEATURES_VERSION

    @property
    def first_date(self) -> date:
        """Premier match joué."""
        return pd.Timestamp(self.frame["match_date"].min()).date()

    @property
    def last_date(self) -> date:
        """Dernier match joué : la campagne ne voit rien au-delà."""
        return pd.Timestamp(self.frame["match_date"].max()).date()


class EmptyDatasetError(RuntimeError):
    """Aucun match joué en base : lancer l'ingestion avant l'entraînement."""


class DatasetBuilder:
    """Charge les matchs validés et rejoue l'historique pour calculer les features."""

    def __init__(self, source: MatchSource, features: FeatureConfig) -> None:
        self._source = source
        self._features = features

    def build(self) -> Dataset:
        """Features + label de chaque match joué ; échoue clairement si la base est vide."""
        matches = self._source.matches_frame()
        started = time.perf_counter()
        frame = FeatureBuilder(self._features).build(matches) if not matches.empty else pd.DataFrame()
        if frame.empty:
            msg = "aucun match joué en base : lancer `make ingest` avant `make train`"
            raise EmptyDatasetError(msg)
        versions: list[str] = (
            [str(v) for v in matches["dataset_version"].dropna().unique()] if "dataset_version" in matches else []
        )
        return Dataset(
            frame=frame,
            matches_loaded=len(matches),
            dataset_versions=tuple(sorted(versions)),
            lots=self._source.lots_summary(),
            build_seconds=time.perf_counter() - started,
        )
