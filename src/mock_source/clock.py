"""Horloge simulée : le « présent » du simulateur, réglable et accélérable.

- `set(date=…)` place l'horloge à une date ; `advance(days=…)` la fait avancer ;
- `speed` (jours simulés par minute réelle) la fait avancer toute seule ; 0 = horloge arrêtée (défaut, déterministe).

La granularité exposée est le **jour** : à la date D, tous les matchs datés avant D sont terminés, ceux du jour D
et après sont à venir.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True, slots=True)
class ClockSnapshot:
    """État de l'horloge à un instant."""

    now: date
    speed: float
    lower: date
    upper: date


class ClockOutOfRangeError(ValueError):
    """Date demandée hors de la fenêtre de rejeu."""


class SimulatedClock:
    """Horloge thread-safe ; la source de temps réel est injectable pour les tests."""

    def __init__(
        self,
        start: date,
        lower: date,
        upper: date,
        speed: float = 0.0,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        if not lower <= upper:
            msg = "lower doit précéder upper"
            raise ValueError(msg)
        self._lower, self._upper = lower, upper
        self._time = time_source
        self._lock = threading.Lock()
        self._anchor_date = self._checked(start)
        self._anchor_real = self._time()
        self._speed = self._checked_speed(speed)

    def now(self) -> date:
        """Date simulée courante."""
        with self._lock:
            return self._current()

    def snapshot(self) -> ClockSnapshot:
        """État complet de l'horloge."""
        with self._lock:
            return ClockSnapshot(self._current(), self._speed, self._lower, self._upper)

    def set(self, when: date | None = None, speed: float | None = None) -> ClockSnapshot:
        """Place l'horloge à une date et/ou change sa vitesse."""
        with self._lock:
            anchor = self._checked(when) if when is not None else self._current()
            self._anchor_date, self._anchor_real = anchor, self._time()
            if speed is not None:
                self._speed = self._checked_speed(speed)
            return ClockSnapshot(self._current(), self._speed, self._lower, self._upper)

    def advance(self, days: int) -> tuple[date, date]:
        """Avance de `days` jours ; renvoie (ancienne date, nouvelle date)."""
        if days < 1:
            msg = "days doit être >= 1"
            raise ValueError(msg)
        with self._lock:
            previous = self._current()
            self._anchor_date = self._checked(previous + timedelta(days=days))
            self._anchor_real = self._time()
            return previous, self._anchor_date

    def _current(self) -> date:
        elapsed_minutes = (self._time() - self._anchor_real) / SECONDS_PER_MINUTE
        current = self._anchor_date + timedelta(days=int(elapsed_minutes * self._speed))
        return min(current, self._upper)

    def _checked(self, when: date) -> date:
        if not self._lower <= when <= self._upper:
            msg = f"{when} hors de la fenêtre de rejeu [{self._lower} ; {self._upper}]"
            raise ClockOutOfRangeError(msg)
        return when

    @staticmethod
    def _checked_speed(speed: float) -> float:
        if not 0.0 <= speed <= 1000.0:
            msg = "speed doit être entre 0 et 1000 jours simulés par minute"
            raise ValueError(msg)
        return speed
