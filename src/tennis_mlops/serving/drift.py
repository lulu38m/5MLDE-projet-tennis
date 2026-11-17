"""Dérive des entrées : les matchs qu'on prédit ressemblent-ils encore à ceux sur lesquels le modèle a appris ?

Pour chaque feature suivie, on compare les dernières prédictions servies (fenêtre glissante) à la période
d'entraînement du modèle servi (les deux années avant `data_window_end`, tag de la version) par le **PSI**
(population stability index) sur des déciles de la référence, avec les seuils de la dérive entre campagnes
(`training.drift`) : moins de 0,1, stable ; 0,1 à 0,2, à surveiller ; au-delà, la distribution a changé et le modèle
extrapole.

Les features sont des écarts entre joueur 1 et joueur 2 : leur signe dépend de l'ordre des joueurs, qui n'est pas
le même à l'entraînement (tiré au hasard) et dans un tableau. On compare donc leur **valeur absolue** (l'écart de
niveau entre les deux joueurs), seule chose qui a un sens pour la dérive ; `elo_win_prob` devient |p - 0,5|.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tennis_mlops.training import drift as training_drift

DRIFT_FEATURES = ("rank_log_ratio", "points_log_ratio", "elo_diff", "surface_elo_diff", "elo_win_prob",
                  "form_diff", "experience_diff", "age_diff", "rest_days_diff", "h2h_diff")  # fmt: skip
CENTERED = {"elo_win_prob": 0.5}
Array = npt.NDArray[np.float64]
Reference = tuple[dict[str, Array], tuple[date | None, date | None]]


@dataclass(frozen=True)
class DriftReport:
    """PSI de chaque feature suivie, et sur quoi il porte."""

    psi: dict[str, float]
    sample: int
    reference_rows: int
    reference_from: date | None
    reference_to: date | None

    def as_dict(self) -> dict[str, Any]:
        """Pour `GET /model`."""
        return {"psi": {k: round(v, 4) for k, v in self.psi.items()}, "sample": self.sample,
                "reference_rows": self.reference_rows, "reference_from": _iso(self.reference_from),
                "reference_to": _iso(self.reference_to)}  # fmt: skip


def magnitude(feature: str, values: Any) -> Array:  # noqa: ANN401
    """L'écart de niveau, indépendant de l'ordre des joueurs."""
    array = np.asarray(values, dtype=np.float64)
    return np.asarray(np.abs(array - CENTERED.get(feature, 0.0)), dtype=np.float64)


def psi(reference: Array, recent: Array, bins: int = 10) -> float:
    """Population stability index de `recent` contre `reference`, sur les quantiles de la référence.

    Même calcul et mêmes seuils que la dérive entre deux campagnes d'entraînement (`training.drift`).
    """
    reference, recent = reference[np.isfinite(reference)], recent[np.isfinite(recent)]
    if len(reference) < bins or len(recent) == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # feature quasi constante : pas de dérive mesurable
        return 0.0
    return training_drift.psi(training_drift.proportions(reference, edges), training_drift.proportions(recent, edges))


@dataclass
class DriftMonitor:
    """Les dernières features servies, et leur comparaison à la période d'entraînement."""

    window: int = 500
    reference_days: int = 730
    bins: int = 10
    _recent: deque[dict[str, float]] = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _cache: tuple[int, date | None, Reference] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.window)

    def observe(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Ajoute les features d'un appel (thread-safe)."""
        kept = [{f: float(r[f]) for f in DRIFT_FEATURES if r.get(f) is not None} for r in rows]
        with self._lock:
            self._recent.extend(kept)

    def report(self, history: pd.DataFrame | None, data_end: date | None) -> DriftReport:
        """PSI de chaque feature : fenêtre récente contre les `reference_days` jours avant `data_end`."""
        with self._lock:
            recent = list(self._recent)
        reference, period = self._reference(history, data_end)
        scores = {}
        for feature in DRIFT_FEATURES:
            values = [r[feature] for r in recent if feature in r]
            if feature in reference and values:
                scores[feature] = psi(reference[feature], magnitude(feature, values), self.bins)
        rows = len(next(iter(reference.values()))) if reference else 0
        return DriftReport(scores, len(recent), rows, *period)

    def _reference(self, history: pd.DataFrame | None, data_end: date | None) -> Reference:
        if history is None or history.empty:
            return {}, (None, None)
        key = id(history)
        if self._cache is not None and self._cache[0] == key and self._cache[1] == data_end:
            return self._cache[2]
        days = pd.to_datetime(history["match_date"]).dt.date
        end = data_end or days.max()
        start = end - timedelta(days=self.reference_days)
        window = history[(days > start) & (days <= end)]
        reference = {f: magnitude(f, window[f]) for f in DRIFT_FEATURES if f in window}
        self._cache = (key, data_end, (reference, (start, end)))
        return reference, (start, end)


def _iso(day: date | None) -> str | None:
    return day.isoformat() if day else None
