"""Découpage **temporel** : on apprend sur le passé, on choisit sur la période suivante, on juge sur la plus récente.

```
train_start ─── entraînement ───┤ validation (choix) ├─ test (bilan) ─┤ dernier match
```

Un découpage aléatoire mélangerait des matchs de 2014 dans l'entraînement d'un modèle évalué sur 2013 : les scores
seraient flatteurs et faux. Ici, chaque partition est strictement postérieure à la précédente.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from tennis_mlops.features import FEATURE_COLUMNS, LABEL
from tennis_mlops.training.config import SplitConfig

PARTS = ("train", "validation", "test")


class InsufficientDataError(RuntimeError):
    """Une partition est trop petite pour entraîner ou juger un modèle de façon fiable."""


@dataclass(frozen=True)
class Split:
    """Les trois partitions et leurs bornes (incluses au début, exclues à la fin)."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_start: date
    validation_start: date
    test_start: date
    end: date

    def part(self, name: str) -> pd.DataFrame:
        """Une partition par son nom (`train`, `validation`, `test`)."""
        frame: pd.DataFrame = getattr(self, name)
        return frame

    @staticmethod
    def features(frame: pd.DataFrame) -> pd.DataFrame:
        """Les colonnes d'entrée du modèle."""
        return frame[list(FEATURE_COLUMNS)]

    @staticmethod
    def labels(frame: pd.DataFrame) -> pd.Series[int]:
        """1 si le joueur 1 a gagné."""
        series: pd.Series[int] = frame[LABEL]
        return series

    def describe(self) -> dict[str, str | int]:
        """Bornes et tailles, pour les paramètres MLflow."""
        return {
            "split.train_start": self.train_start.isoformat(),
            "split.validation_start": self.validation_start.isoformat(),
            "split.test_start": self.test_start.isoformat(),
            "split.end": self.end.isoformat(),
            **{f"split.{name}_rows": len(self.part(name)) for name in PARTS},
        }


class TemporalSplitter:
    """Calcule les bornes à partir du dernier match disponible."""

    def __init__(self, config: SplitConfig) -> None:
        self._config = config

    def split(self, frame: pd.DataFrame) -> Split:
        """Partitions train / validation / test ; refuse une partition sous `min_rows`."""
        dates = pd.to_datetime(frame["match_date"])
        end = dates.max().date()
        test_start = (pd.Timestamp(end) - pd.DateOffset(months=self._config.test_months)).date() + timedelta(days=1)
        validation_start = (pd.Timestamp(test_start) - pd.DateOffset(months=self._config.validation_months)).date()
        train_start = self._config.train_start
        bounds = {
            "train": (train_start, validation_start),
            "validation": (validation_start, test_start),
            "test": (test_start, end + timedelta(days=1)),
        }
        parts = {
            name: frame[(dates >= pd.Timestamp(lo)) & (dates < pd.Timestamp(hi))].reset_index(drop=True)
            for name, (lo, hi) in bounds.items()
        }
        too_small = {n: len(p) for n, p in parts.items() if len(p) < self._config.min_rows}
        if too_small:
            detail = ", ".join(f"{n} = {size}" for n, size in too_small.items())
            msg = (
                f"partition(s) trop petite(s) ({detail} < {self._config.min_rows} matchs) avec des données jusqu'au "
                f"{end} : ingérer davantage ou réduire les fenêtres de configs/train.yaml"
            )
            raise InsufficientDataError(msg)
        return Split(
            train=parts["train"],
            validation=parts["validation"],
            test=parts["test"],
            train_start=train_start,
            validation_start=validation_start,
            test_start=test_start,
            end=end,
        )
