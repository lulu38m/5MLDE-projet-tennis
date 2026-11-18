"""Faire avancer le temps simulé de `mock_source` (`POST /admin/clock/advance`) : le moteur de la démo.

Seule écriture du projet vers la source, et seulement sur son API d'administration (jamais touchée par le chaos).

L'horloge de la source vit en mémoire : un redémarrage de `mock_source` (`make down`, une image reconstruite) la
renvoie au début du rejeu alors que la base a déjà ingéré des années. `catch_up` la remet à la date jusqu'à laquelle
tout est ingéré (le watermark), avant chaque pas : sans cela, la démo rejouerait des tournois déjà connus et le
scoring ne trouverait aucun match à venir.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import httpx

from tennis_mlops.data.errors import SourceUnavailableError
from tennis_mlops.ingestion.client import TIMEOUT_ERRORS, TRANSPORT_ERRORS, HttpResponse

ADVANCE_PATH = "/admin/clock/advance"
CLOCK_PATH = "/admin/clock"
OUT_OF_RANGE = 422  # l'horloge est déjà au bout du rejeu


class ReplayFinishedError(RuntimeError):
    """L'horloge est au bout du rejeu (`simulation.replay_end`) : il n'y a plus rien à avancer."""


class HttpPoster(Protocol):
    """Tout client HTTP capable d'un GET et d'un POST JSON (httpx.Client, TestClient de Starlette…)."""

    def post(self, url: str, *, json: Mapping[str, Any]) -> HttpResponse:
        """Requête POST."""
        ...

    def get(self, url: str) -> HttpResponse:
        """Requête GET."""
        ...


@dataclass(frozen=True)
class ClockStep:
    """Ce qu'a fait un pas de simulation."""

    previous: date
    now: date
    new_matches: int
    tournaments: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Une ligne pour les logs."""
        return (
            f"horloge {self.previous} → {self.now} : {self.new_matches} match(s) terminé(s), "
            f"{len(self.tournaments)} tournoi(s) fini(s)"
        )


class SimulationClock:
    """`POST /admin/clock/advance` ; le client HTTP est injectable pour les tests."""

    def __init__(self, base_url: str, timeout_s: float, http: HttpPoster | None = None) -> None:
        self._owned = None if http is not None else httpx.Client(base_url=base_url, timeout=timeout_s)
        self._http: HttpPoster = http if http is not None else self._owned  # type: ignore[assignment]

    def close(self) -> None:
        """Ferme la connexion si ce client l'a ouverte."""
        if self._owned is not None:
            self._owned.close()

    def catch_up(self, day: date) -> date | None:
        """Ramène l'horloge à `day` si elle est en retard.

        Renvoie la date où elle était (None si elle était à l'heure) ; `SourceUnavailableError` sur panne.
        """
        try:
            current = self._http.get(CLOCK_PATH)
            if current.status_code >= 300:
                raise SourceUnavailableError(f"GET {CLOCK_PATH} : HTTP {current.status_code}")
            now = date.fromisoformat(current.json()["now"])
            if now >= day:
                return None
            response = self._http.post(CLOCK_PATH, json={"date": day.isoformat()})
        except (*TIMEOUT_ERRORS, *TRANSPORT_ERRORS) as exc:
            msg = f"{CLOCK_PATH} : source injoignable ({type(exc).__name__})"
            raise SourceUnavailableError(msg) from exc
        if response.status_code >= 300:
            raise SourceUnavailableError(f"POST {CLOCK_PATH} : HTTP {response.status_code} ({response.text[:200]})")
        return now

    def advance(self, days: int) -> ClockStep:
        """Avance de `days` jours ; `ReplayFinishedError` au bout du rejeu, `SourceUnavailableError` sur panne."""
        try:
            response = self._http.post(ADVANCE_PATH, json={"days": days})
        except (*TIMEOUT_ERRORS, *TRANSPORT_ERRORS) as exc:
            msg = f"POST {ADVANCE_PATH} : source injoignable ({type(exc).__name__})"
            raise SourceUnavailableError(msg) from exc
        if response.status_code == OUT_OF_RANGE:
            raise ReplayFinishedError(response.text[:200])
        if response.status_code >= 300:
            msg = f"POST {ADVANCE_PATH} : HTTP {response.status_code}"
            raise SourceUnavailableError(msg)
        data = response.json()
        return ClockStep(
            previous=date.fromisoformat(data["previous"]),
            now=date.fromisoformat(data["now"]),
            new_matches=int(data["newly_completed_matches"]),
            tournaments=list(data["newly_completed_tournaments"]),
        )
