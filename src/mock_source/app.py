"""API FastAPI de rejeu ATP.

Toutes les réponses portent `X-Simulated-Now` et `X-Dataset-Version`. Toute date demandée après l'horloge simulée
est refusée (`400`) : l'API ne peut pas révéler le futur, quel que soit le paramètre utilisé.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import date, timedelta
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mock_source.chaos import ChaosController, ChaosMode, ChaosSettings, break_schema, corrupt_payload
from mock_source.clock import ClockOutOfRangeError, SimulatedClock
from mock_source.repository import ReplayRepository
from mock_source.schemas import (
    AdvanceRequest,
    AdvanceResponse,
    ClockRequest,
    ClockState,
    FixturesPage,
    Health,
    PlayerHistory,
    RankingsPage,
    ResultsPage,
    TournamentResults,
    TournamentsPage,
    TournamentStatus,
)

logger = logging.getLogger(__name__)

UNAFFECTED_PREFIXES = ("/admin", "/health", "/docs", "/openapi.json", "/redoc")
SIMULATED_NOW_HEADER = "X-Simulated-Now"
DATASET_VERSION_HEADER = "X-Dataset-Version"
CHAOS_HEADER = "X-Chaos-Mode"


def _not_after_now(value: date, now: date, name: str) -> None:
    if value > now:
        raise HTTPException(400, detail=f"{name}={value} est postérieur à l'horloge simulée ({now})")


def create_app(
    repository: ReplayRepository,
    clock: SimulatedClock,
    chaos: ChaosController | None = None,
    fixture_horizon_days: int = 7,
) -> FastAPI:
    """Construit l'application (dépendances injectées : les tests utilisent de petites données)."""
    chaos = chaos or ChaosController()
    app = FastAPI(
        title="mock_source — rejeu ATP",
        version="1.0.0",
        description="Rejoue l'historique ATP de Jeff Sackmann (CC BY-NC-SA 4.0) selon une horloge simulée.",
    )

    @app.middleware("http")
    async def simulated_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if not request.url.path.startswith(UNAFFECTED_PREFIXES):
            setting = chaos.consume()
            request.state.chaos = setting
            if setting.mode is ChaosMode.UNAVAILABLE:
                response: Response = JSONResponse(
                    {"detail": "Service indisponible (chaos 503)"}, status_code=503, headers={"Retry-After": "1"}
                )
                return _decorate(response, setting.mode)
            if setting.mode is ChaosMode.SLOW:
                await asyncio.sleep(setting.delay_s)
        response = await call_next(request)
        mode = getattr(request.state, "chaos", ChaosSettings()).mode
        return _decorate(response, mode)

    def _decorate(response: Response, mode: ChaosMode) -> Response:
        response.headers[SIMULATED_NOW_HEADER] = clock.now().isoformat()
        response.headers[DATASET_VERSION_HEADER] = repository.dataset_version
        if mode is not ChaosMode.OFF:
            response.headers[CHAOS_HEADER] = mode.value
        return response

    def _payload(request: Request, model: BaseModel) -> JSONResponse:
        """Sérialise la réponse puis applique le chaos éventuel.

        La réponse est renvoyée telle quelle (`JSONResponse`) : FastAPI ne la revalide pas contre `response_model`,
        sinon les incidents `corrupt` et `schema` seraient bloqués avant d'atteindre le client.
        """
        payload: dict[str, Any] = model.model_dump(mode="json")
        mode = getattr(request.state, "chaos", ChaosSettings()).mode
        if mode is ChaosMode.CORRUPT:
            payload = corrupt_payload(payload, clock.now())
        elif mode is ChaosMode.SCHEMA:
            payload = break_schema(payload)
        return JSONResponse(payload)

    # ------------------------------------------------------------------ santé et administration

    @app.get("/health", tags=["admin"])
    def health() -> Health:
        """État du service."""
        return Health(
            status="ok",
            dataset_version=repository.dataset_version,
            simulated_now=clock.now(),
            matches_loaded=repository.match_count,
            chaos_mode=chaos.state().mode.value,
        )

    @app.get("/admin/clock", tags=["admin"])
    def get_clock() -> ClockState:
        """Date simulée et vitesse."""
        s = clock.snapshot()
        return ClockState(now=s.now, speed=s.speed, lower=s.lower, upper=s.upper)

    @app.post("/admin/clock", tags=["admin"])
    def set_clock(body: ClockRequest) -> ClockState:
        """Place l'horloge à une date et/ou change sa vitesse (jours simulés par minute réelle)."""
        if body.date is None and body.speed is None:
            raise HTTPException(422, detail="fournir date et/ou speed")
        try:
            s = clock.set(body.date, body.speed)
        except ClockOutOfRangeError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        logger.info("Horloge réglée : now=%s speed=%s", s.now, s.speed)
        return ClockState(now=s.now, speed=s.speed, lower=s.lower, upper=s.upper)

    @app.post("/admin/clock/advance", tags=["admin"])
    def advance_clock(body: AdvanceRequest) -> AdvanceResponse:
        """Fait avancer le temps et indique ce qui vient de se terminer."""
        try:
            previous, now = clock.advance(body.days)
        except ClockOutOfRangeError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        return AdvanceResponse(
            previous=previous,
            now=now,
            newly_completed_matches=repository.count_completed_between(previous, now),
            newly_completed_tournaments=repository.tournaments_completed_between(previous, now),
        )

    @app.get("/admin/chaos", tags=["admin"])
    def get_chaos() -> ChaosSettings:
        """Mode d'incident courant."""
        return chaos.state()

    @app.post("/admin/chaos", tags=["admin"])
    def set_chaos(body: ChaosSettings) -> ChaosSettings:
        """Active un incident contrôlé (`503`, `slow`, `corrupt`, `schema`) ou le désactive (`off`)."""
        logger.warning("Chaos : %s", body.model_dump())
        return chaos.configure(body)

    # ------------------------------------------------------------------ données

    @app.get("/results", tags=["matchs"], response_model=ResultsPage)
    def results(
        request: Request,
        since: Annotated[date, Query(description="Début de période (inclus), typiquement le watermark")],
        until: Annotated[date | None, Query(description="Fin de période (exclue), au plus now")] = None,
    ) -> JSONResponse:
        """Matchs terminés dans `[since ; until[`."""
        now = clock.now()
        end = until or now
        _not_after_now(end, now, "until")
        if since > end:
            raise HTTPException(400, detail="since doit précéder until")
        rows = repository.results(since, end)
        reason = repository.empty_period_reason(since, end) if not rows else None
        return _payload(request, ResultsPage(since=since, until=end, count=len(rows), results=rows, reason=reason))

    @app.get("/fixtures", tags=["matchs"], response_model=FixturesPage)
    def fixtures(
        request: Request,
        days: Annotated[int | None, Query(ge=1, le=30, description="Horizon en jours")] = None,
    ) -> JSONResponse:
        """Matchs à venir dont les deux joueurs sont connus, sans aucun résultat."""
        now = clock.now()
        horizon = days or fixture_horizon_days
        rows = repository.fixtures(now, horizon)
        reason = repository.empty_period_reason(now, now + timedelta(days=horizon)) if not rows else None
        return _payload(
            request, FixturesPage(now=now, horizon_days=horizon, count=len(rows), fixtures=rows, reason=reason)
        )

    @app.get("/tournaments", tags=["tournois"])
    def tournaments(
        since: Annotated[date | None, Query(description="Date de fin >= since")] = None,
        until: Annotated[date | None, Query(description="Date de fin < until")] = None,
        status: TournamentStatus | None = None,
    ) -> TournamentsPage:
        """Calendrier visible à la date simulée."""
        now = clock.now()
        rows = repository.tournaments(now, since, until, status)
        return TournamentsPage(count=len(rows), tournaments=rows)

    @app.get("/tournaments/{tourney_id}/results", tags=["tournois"], response_model=TournamentResults)
    def tournament_results(request: Request, tourney_id: Annotated[str, Path()]) -> JSONResponse:
        """Un lot d'ingestion : résultats déjà connus d'un tournoi + nombre de matchs attendus."""
        found = repository.tournament(tourney_id, clock.now())
        if found is None:
            raise HTTPException(404, detail=f"tournoi {tourney_id} inconnu ou pas encore annoncé")
        summary, rows = found
        page = TournamentResults(
            tournament=summary, expected_matches=summary.expected_matches, count=len(rows), results=rows
        )
        return _payload(request, page)

    @app.get("/rankings", tags=["joueurs"])
    def rankings(
        date_: Annotated[date | None, Query(alias="date", description="Défaut : date simulée")] = None,
        limit: Annotated[int, Query(ge=1, le=5000)] = 100,
        player_id: int | None = None,
    ) -> RankingsPage:
        """Dernier classement publié à la date demandée (jamais après l'horloge)."""
        now = clock.now()
        requested = date_ or now
        _not_after_now(requested, now, "date")
        ranking_date, rows = repository.rankings(requested, limit, player_id)
        reason = "Aucun classement publié avant cette date" if ranking_date is None else None
        return RankingsPage(
            requested_date=requested, ranking_date=ranking_date, count=len(rows), rankings=rows, reason=reason
        )

    @app.get("/players/{player_id}/history", tags=["joueurs"])
    def player_history(
        player_id: int,
        as_of: Annotated[date, Query(description="Obligatoire : date de référence, au plus now")],
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        weeks: Annotated[int, Query(ge=1, le=520, description="Profondeur de l'historique de classement")] = 52,
    ) -> PlayerHistory:
        """Matchs (strictement avant as_of) et classements (jusqu'à as_of) d'un joueur."""
        _not_after_now(as_of, clock.now(), "as_of")
        history = repository.player_history(player_id, as_of, limit, weeks)
        if history is None:
            raise HTTPException(404, detail=f"joueur {player_id} inconnu")
        return history

    return app
