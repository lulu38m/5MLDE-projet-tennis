"""Client HTTP typé de `mock_source` : une méthode par endpoint, des erreurs qui disent quoi faire.

- `SourceUnavailableError` : panne (connexion, timeout, 5xx, corps illisible) → on peut réessayer.
- `SourceRequestError` : la source répond mais refuse (4xx) ou rompt son contrat d'en-têtes → réessayer est inutile.

Le client ne valide **pas** le contenu des données : il renvoie le JSON tel quel avec ses métadonnées
(`X-Simulated-Now`, `X-Dataset-Version`). Le contrôle du contenu est le rôle de Great Expectations, qui doit voir
exactement ce que la source a envoyé, y compris un schéma cassé.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from types import TracebackType
from typing import Any, Protocol, Self

import httpx
import httpx2

from tennis_mlops.data.errors import SourceRequestError, SourceUnavailableError

SIMULATED_NOW_HEADER = "X-Simulated-Now"
DATASET_VERSION_HEADER = "X-Dataset-Version"
# httpx (client réel) et httpx2 (fork utilisé par le TestClient de Starlette) : mêmes erreurs, deux hiérarchies.
TIMEOUT_ERRORS = (httpx.TimeoutException, httpx2.TimeoutException)
TRANSPORT_ERRORS = (httpx.TransportError, httpx2.TransportError)


class HttpResponse(Protocol):
    """Ce que le client lit d'une réponse HTTP."""

    @property
    def status_code(self) -> int:
        """Code HTTP."""
        ...

    @property
    def text(self) -> str:
        """Corps en texte."""
        ...

    @property
    def headers(self) -> Mapping[str, str]:
        """En-têtes."""
        ...

    def json(self) -> Any:  # noqa: ANN401 — JSON arbitraire, contrôlé ensuite par Great Expectations
        """Corps décodé."""
        ...


class HttpGetter(Protocol):
    """Tout client HTTP capable d'un GET (httpx.Client, TestClient de Starlette…)."""

    def get(self, url: str, *, params: Mapping[str, str] | None = None) -> HttpResponse:
        """Requête GET."""
        ...


@dataclass(frozen=True, slots=True)
class SourcePayload:
    """Réponse brute d'un endpoint, avec le contexte de la source au moment de l'appel."""

    endpoint: str
    data: dict[str, Any]
    simulated_now: date
    dataset_version: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MockSourceClient:
    """Accès en lecture à l'API de rejeu (le client HTTP est injectable pour les tests)."""

    def __init__(self, base_url: str, timeout_s: float, http: HttpGetter | None = None) -> None:
        self._owned = None if http is not None else httpx.Client(base_url=base_url, timeout=timeout_s)
        self._http: HttpGetter = http if http is not None else self._owned  # type: ignore[assignment]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        """Ferme la connexion si ce client l'a ouverte."""
        if self._owned is not None:
            self._owned.close()

    # ------------------------------------------------------------------ endpoints

    def clock(self) -> SourcePayload:
        """`GET /admin/clock` : date simulée de la source."""
        return self._get("/admin/clock")

    def tournaments(self, since: date, until: date, status: str = "completed") -> SourcePayload:
        """`GET /tournaments` : tournois dont la date de fin est dans `[since ; until[`."""
        return self._get("/tournaments", {"since": since.isoformat(), "until": until.isoformat(), "status": status})

    def tournament_results(self, tourney_id: str) -> SourcePayload:
        """`GET /tournaments/{id}/results` : un lot (résultats + `expected_matches`)."""
        return self._get(f"/tournaments/{tourney_id}/results")

    def fixtures(self, days: int | None = None) -> SourcePayload:
        """`GET /fixtures` : matchs à venir dont les deux joueurs sont connus (scoring, phase 7)."""
        return self._get("/fixtures", {"days": str(days)} if days else None)

    def results(self, since: date, until: date) -> SourcePayload:
        """`GET /results` : matchs terminés dans `[since ; until[` (amorçage de l'historique)."""
        return self._get("/results", {"since": since.isoformat(), "until": until.isoformat()})

    # ------------------------------------------------------------------ transport

    def _get(self, path: str, params: dict[str, str] | None = None) -> SourcePayload:
        try:
            response = self._http.get(path, params=params)
        except TIMEOUT_ERRORS as exc:
            msg = f"GET {path} : délai dépassé ({type(exc).__name__})"
            raise SourceUnavailableError(msg) from exc
        except TRANSPORT_ERRORS as exc:
            msg = f"GET {path} : source injoignable ({type(exc).__name__})"
            raise SourceUnavailableError(msg) from exc
        if response.status_code >= 500:
            msg = f"GET {path} : HTTP {response.status_code}"
            raise SourceUnavailableError(msg)
        if response.status_code >= 400:
            msg = f"GET {path} : HTTP {response.status_code} — {response.text[:200]}"
            raise SourceRequestError(msg)
        try:
            data = response.json()
        except ValueError as exc:
            msg = f"GET {path} : corps de réponse illisible"
            raise SourceUnavailableError(msg) from exc
        return SourcePayload(
            endpoint=path,
            data=data,
            simulated_now=self._simulated_now(response, path),
            dataset_version=response.headers.get(DATASET_VERSION_HEADER, "inconnue"),
        )

    @staticmethod
    def _simulated_now(response: HttpResponse, path: str) -> date:
        raw = response.headers.get(SIMULATED_NOW_HEADER)
        try:
            return date.fromisoformat(raw or "")
        except ValueError as exc:
            msg = f"GET {path} : en-tête {SIMULATED_NOW_HEADER} absent ou invalide ({raw!r})"
            raise SourceRequestError(msg) from exc
