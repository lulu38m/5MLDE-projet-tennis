"""API de prédiction : probabilité qu'un joueur gagne un match ATP, servie par la version `production` du registre.

| Endpoint | Rôle |
|---|---|
| `POST /predict` | un match → probabilité de chaque joueur, version, `request_id` |
| `POST /predict/batch` | jusqu'à `api.max_batch` matchs en un appel (scoring des matchs de la semaine) |
| `POST /feedback` | le vrai résultat d'un match → prédictions jugées (accuracy, log loss en ligne) |
| `GET /model` | version servie, provenance, état du registre, historique, qualité en ligne |
| `GET /health` | le processus répond (liveness) |
| `GET /ready` | peut-on prédire : `ready`, `degraded` (MLflow ou base injoignable, cache servi) ou `503` |
| `POST /admin/reload` | relire tout de suite l'alias et l'historique |
| `GET /metrics` | métriques Prometheus |
| `GET /players`, `/players/{id}/stats`, `/h2h` | joueurs, statistiques, face-à-face (historique validé seulement) |
| `GET /fixtures/predictions` | matchs prédits à l'avance par le scoring, et leur résultat une fois validé |

Chaque réponse porte `X-Request-ID` (repris de la requête s'il est fourni) ; les prédictions portent aussi
`X-Model-Version`. Le client se présente avec `X-Client` (`ui`, `scoring`, `feedback`, `locust`…) : il est gardé
avec chaque trace, pour distinguer les usages dans Grafana. Toute réponse 4xx/5xx laisse une ligne dans `errors`.
"""

from __future__ import annotations

import logging
import re
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from typing import Annotated, Any

from fastapi import FastAPI, Header, Path, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from tennis_mlops.api.metrics import ApiMetrics
from tennis_mlops.serving.persistence import to_json
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
from tennis_mlops.serving.service import (
    InvalidMatchError,
    PredictionService,
    ServiceUnavailableError,
    UnknownMatchError,
    UnknownPlayerError,
)

logger = logging.getLogger(__name__)
REQUEST_ID = re.compile(r"^[\w.-]{1,64}$")
CLIENT = re.compile(r"[^\w.-]")
UNTRACED = {"/metrics", "/health"}  # sondes fréquentes : comptées dans Prometheus, pas écrites en base


def create_app(service: PredictionService, admin_token: str | None = None, manage_service: bool = True) -> FastAPI:
    """L'application ; `manage_service` démarre et arrête le service avec elle (désactivé par certains tests)."""
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:  # noqa: ANN401
        if manage_service:
            service.start()
        yield
        if manage_service:
            service.stop()

    app = FastAPI(title="tennis-mlops · API de prédiction", version="1.0", description=__doc__, lifespan=lifespan)
    app.state.service = service
    app.state.metrics = metrics

    # ------------------------------------------------------------------ traçage de chaque requête

    @app.middleware("http")
    async def trace(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        started = time.perf_counter()
        received_at = datetime.now(UTC)
        given = request.headers.get("x-request-id", "")
        request.state.request_id = given if REQUEST_ID.match(given) else uuid.uuid4().hex
        request.state.client = CLIENT.sub("", request.headers.get("x-client", ""))[:32] or None
        request.state.started = started
        request.state.predictions = 0
        try:
            response = await call_next(request)
        except Exception:
            observe(request, received_at, started, 500, None)  # l'erreur elle-même est tracée par `internal`
            raise
        response.headers["X-Request-ID"] = request.state.request_id
        observe(request, received_at, started, response.status_code, response.headers.get("x-model-version"))
        return response

    def observe(request: Request, received_at: datetime, started: float, status: int, version: str | None) -> None:
        duration = time.perf_counter() - started
        path = getattr(request.scope.get("route"), "path", None) or "inconnue"
        metrics.requests.labels(request.method, path, str(status)).inc()
        metrics.latency.labels(path).observe(duration)
        if path not in UNTRACED:
            service.writer.put("api_requests", {
                "request_id": request.state.request_id, "received_at": received_at, "method": request.method,
                "path": path, "status_code": status, "duration_ms": round(duration * 1000, 3),
                "client": request.state.client, "model_version": version, "predictions": request.state.predictions,
            })  # fmt: skip

    def fail(request: Request, status: int, error_type: str, message: str, detail: object = None) -> JSONResponse:
        """Réponse d'erreur + ligne dans `errors` + compteur Prometheus."""
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        metrics.errors.labels(error_type).inc()
        route = request.scope.get("route")
        service.writer.put("errors", {
            "occurred_at": datetime.now(UTC), "request_id": request_id, "method": request.method,
            "path": getattr(route, "path", None) or request.url.path[:64], "status_code": status,
            "error_type": error_type, "message": message[:2000], "detail": to_json(jsonable_encoder(detail)),
            "client": getattr(request.state, "client", None),
        })  # fmt: skip
        body: dict[str, Any] = {"error": error_type, "detail": message, "request_id": request_id}
        if detail is not None:
            body["errors"] = jsonable_encoder(detail)
        return JSONResponse(status_code=status, content=body)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        first = errors[0] if errors else {"loc": [], "msg": "requête invalide"}
        where = ".".join(str(p) for p in first["loc"] if p != "body") or "requête"
        message = f"{len(errors)} champ(s) invalide(s) ; {where} : {first['msg']}"
        return fail(request, 422, "validation", message, errors)

    @app.exception_handler(InvalidMatchError)
    async def invalid_match(request: Request, exc: InvalidMatchError) -> JSONResponse:
        return fail(request, 422, "invalid_match", str(exc), {"index": exc.index} if exc.index is not None else None)

    @app.exception_handler(UnknownMatchError)
    async def unknown_match(request: Request, exc: UnknownMatchError) -> JSONResponse:
        return fail(request, 422, "inconsistent_feedback", str(exc))

    @app.exception_handler(UnknownPlayerError)
    async def unknown_player(request: Request, exc: UnknownPlayerError) -> JSONResponse:
        return fail(request, 404, "unknown_player", str(exc))

    @app.exception_handler(ServiceUnavailableError)
    async def unavailable(request: Request, exc: ServiceUnavailableError) -> JSONResponse:
        return fail(request, 503, "unavailable", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return fail(request, exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    @app.exception_handler(Exception)
    async def internal(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("erreur interne (requête %s)", getattr(request.state, "request_id", "?"))
        frames = traceback.extract_tb(exc.__traceback__)[-3:]
        where = [f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}" for f in frames]
        response = fail(request, 500, "internal", f"{type(exc).__name__} : {exc}", {"traceback": where})
        response.headers["X-Request-ID"] = getattr(request.state, "request_id", "")
        return response

    # ------------------------------------------------------------------ endpoints

    @app.get("/health", tags=["état"])
    def health() -> dict[str, str]:
        """Le processus répond (ne dit rien du modèle : voir /ready)."""
        return {"status": "ok"}

    @app.get("/ready", tags=["état"])
    def ready(response: Response) -> dict[str, Any]:
        """`ready` ou `degraded` (200, on sert) ; `not_ready` (503 : pas de modèle ou pas d'historique)."""
        readiness = service.readiness()
        if not readiness.serving:
            response.status_code = 503
        served = service.model.served
        if served is not None:
            response.headers["X-Model-Version"] = served.version
        return {"status": readiness.status, "checks": readiness.checks, "reasons": readiness.reasons}

    @app.get("/model", tags=["état"])
    def model(response: Response) -> dict[str, Any]:
        """Version servie, provenance, état du registre et de l'historique, qualité en ligne."""
        info = service.model_info()
        if info["served"]:
            response.headers["X-Model-Version"] = info["served"]["version"]
        return info

    @app.post("/predict", tags=["prédiction"])
    def predict(body: MatchRequest, request: Request, response: Response) -> PredictResponse:
        """Probabilité de victoire de chaque joueur pour un match à venir."""
        batch = service.predict([body])
        service.record(batch, [body], request.state.request_id, request.state.client)
        _count(request, response, batch.served.version, 1)
        prediction = batch.predictions[0]
        return PredictResponse(**prediction.model_dump(), request_id=request.state.request_id, model=batch.model,
                               latency_ms=_elapsed(request))  # fmt: skip

    @app.post("/predict/batch", tags=["prédiction"])
    def predict_batch(body: BatchRequest, request: Request, response: Response) -> BatchResponse:
        """Plusieurs matchs en un appel ; tous refusés si l'un d'eux ne peut pas être prédit (son index est donné)."""
        batch = service.predict(body.matches)
        service.record(batch, body.matches, request.state.request_id, request.state.client)
        _count(request, response, batch.served.version, len(body.matches))
        return BatchResponse(request_id=request.state.request_id, model=batch.model, count=len(batch.predictions),
                             latency_ms=_elapsed(request), predictions=batch.predictions)  # fmt: skip

    @app.post("/feedback", tags=["prédiction"])
    def feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
        """Le vrai résultat d'un match : enregistré, et chaque prédiction de ce match est jugée."""
        updated, outcomes = service.feedback(body.match_id, body.winner_id, request.state.request_id,
                                             request.state.client)  # fmt: skip
        for outcome in outcomes:
            metrics.feedbacks.labels(outcome.model_version, str(outcome.correct).lower()).inc()
        return FeedbackResponse(request_id=request.state.request_id, match_id=body.match_id, winner_id=body.winner_id,
                                updated=updated, predictions=outcomes)  # fmt: skip

    @app.post("/admin/reload", tags=["administration"])
    def reload(x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
        """Relit tout de suite l'alias `production` et l'historique (après un rollback, pour ne pas attendre)."""
        if admin_token and x_admin_token != admin_token:
            raise StarletteHTTPException(
                status_code=403, detail="jeton d'administration absent ou faux (X-Admin-Token)"
            )
        return service.reload()

    # ------------------------------------------------------------------ consultation (interface de test)

    @app.get("/players", tags=["consultation"])
    def players(
        search: Annotated[str | None, Query(max_length=64, description="Nom (sans accents) ou identifiant")] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 20,
    ) -> PlayersResponse:
        """Recherche de joueurs ; sans texte, les joueurs classés ayant joué dans l'année, par classement."""
        return service.search_players(search, limit)

    @app.get("/players/{player_id}/stats", tags=["consultation"])
    def player_stats(
        player_id: Annotated[int, Path(gt=0)],
        as_of: Annotated[date | None, Query(description="Défaut et maximum : dernière journée de l'historique")] = None,
    ) -> PlayerStats:
        """Classement dans le temps, bilans par surface et niveau, forme, derniers résultats."""
        return service.player_stats(player_id, as_of)

    @app.get("/h2h", tags=["consultation"])
    def h2h(
        p1: Annotated[int, Query(gt=0)], p2: Annotated[int, Query(gt=0)], as_of: Annotated[date | None, Query()] = None
    ) -> H2HResponse:
        """Rencontres des deux joueurs dans l'historique validé, bilan par surface."""
        return service.head_to_head(p1, p2, as_of)

    @app.get("/fixtures/predictions", tags=["consultation"])
    def fixture_predictions(
        days: Annotated[int, Query(ge=1, le=120, description="Matchs datés depuis N jours avant l'horloge")] = 14,
        client: Annotated[str, Query(max_length=32, description="Auteur des prédictions ; vide = tous")] = "scoring",
    ) -> FixturesResponse:
        """Matchs prédits à l'avance, et pour ceux qui sont joués : le vrai résultat et si le modèle avait raison."""
        return service.fixture_predictions(days, client)

    @app.get("/metrics", tags=["état"], include_in_schema=False)
    def prometheus() -> Response:
        """Métriques Prometheus."""
        return Response(metrics.render(service), media_type="text/plain; version=0.0.4; charset=utf-8")

    def _count(request: Request, response: Response, version: str, n: int) -> None:
        request.state.predictions = n
        response.headers["X-Model-Version"] = version
        metrics.predictions.labels(version, request.state.client or "inconnu").inc(n)
        metrics.batch_size.observe(n)

    return app


def _elapsed(request: Request) -> float:
    started: float = request.state.started
    return round((time.perf_counter() - started) * 1000, 3)
