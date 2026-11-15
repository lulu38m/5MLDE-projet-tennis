"""Client typé de l'API de prédiction, pour les flows (scoring, feedback) et l'interface de test (phase 8).

Chaque appel se présente avec `X-Client` : les traces de l'API distinguent ainsi le scoring, le feedback, l'interface
et les tests de charge. Les erreurs de l'API sont rendues lisibles (`ApiError` : code, type d'erreur, message).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, Self

import httpx

from tennis_mlops.serving.schemas import (
    BatchRequest,
    BatchResponse,
    FeedbackRequest,
    FeedbackResponse,
    FixturesResponse,
    H2HResponse,
    MatchRequest,
    PlayersResponse,
    PlayerStats,
    PredictResponse,
)


class ApiError(RuntimeError):
    """L'API a répondu par une erreur, ou n'a pas répondu."""

    def __init__(self, status: int | None, kind: str, message: str) -> None:
        super().__init__(f"API {status or 'injoignable'} {kind} : {message}")
        self.status = status
        self.kind = kind
        self.message = message


class ApiClient:
    """Prédiction, feedback, état du service, et consultation des joueurs (interface de test)."""

    def __init__(self, base_url: str, client_name: str, timeout_s: float = 30.0, http: object = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._owned = None if http is not None else httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)
        self._http: Any = http if http is not None else self._owned  # httpx.Client, ou le TestClient des tests
        self.headers = {"X-Client": client_name}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Ferme la connexion si ce client l'a ouverte."""
        if self._owned is not None:
            self._owned.close()

    def predict(self, match: MatchRequest) -> PredictResponse:
        """Prédit un match."""
        body = match.model_dump(mode="json", exclude_none=True)
        return PredictResponse.model_validate(self._call("POST", "/predict", json=body))

    def predict_batch(self, matches: Sequence[MatchRequest]) -> BatchResponse:
        """Prédit plusieurs matchs en un appel."""
        body = BatchRequest(matches=list(matches)).model_dump(mode="json", exclude_none=True)
        return BatchResponse.model_validate(self._call("POST", "/predict/batch", json=body))

    def feedback(self, match_id: str, winner_id: int) -> FeedbackResponse:
        """Envoie le vrai résultat d'un match."""
        body = FeedbackRequest(match_id=match_id, winner_id=winner_id).model_dump(mode="json")
        return FeedbackResponse.model_validate(self._call("POST", "/feedback", json=body))

    def model(self) -> dict[str, Any]:
        """Version servie et état de l'historique."""
        return dict(self._call("GET", "/model"))

    def ready(self) -> dict[str, Any]:
        """`/ready`, sans lever si le service n'est pas prêt (503)."""
        return dict(self._call("GET", "/ready", accept=(200, 503)))

    def players(self, search: str | None = None, limit: int = 20) -> PlayersResponse:
        """Recherche de joueurs (sans texte : les joueurs classés actifs)."""
        params: dict[str, Any] = {"limit": limit} | ({"search": search} if search else {})
        return PlayersResponse.model_validate(self._call("GET", "/players", params=params))

    def player_stats(self, player_id: int, as_of: date | None = None) -> PlayerStats:
        """Statistiques d'un joueur jusqu'à `as_of` (défaut : dernière journée de l'historique)."""
        params = {"as_of": as_of.isoformat()} if as_of else {}
        return PlayerStats.model_validate(self._call("GET", f"/players/{player_id}/stats", params=params))

    def h2h(self, p1: int, p2: int) -> H2HResponse:
        """Face-à-face de deux joueurs."""
        return H2HResponse.model_validate(self._call("GET", "/h2h", params={"p1": p1, "p2": p2}))

    def fixture_predictions(self, days: int = 14, client: str = "scoring") -> FixturesResponse:
        """Matchs prédits à l'avance et leur résultat."""
        params = {"days": days, "client": client}
        return FixturesResponse.model_validate(self._call("GET", "/fixtures/predictions", params=params))

    def _call(self, method: str, path: str, accept: tuple[int, ...] = (200,), **kwargs: Any) -> Any:  # noqa: ANN401
        try:
            response = self._http.request(method, path, headers=self.headers, **kwargs)
        except Exception as exc:  # httpx ou httpx2 selon le client : toutes deux signifient « pas de réponse »
            raise ApiError(None, "unreachable", f"{method} {path} : {type(exc).__name__}") from exc
        if response.status_code not in accept:
            try:
                data = response.json()
            except ValueError:
                data = {}
            raise ApiError(
                response.status_code, str(data.get("error", "http")), str(data.get("detail", response.text[:200]))
            )
        return response.json()
