"""Flow Prefect de scoring : prédire, via l'API, les matchs de la semaine qui vient.

```
GET /fixtures (mock_source) ─► déjà prédits ? (table predictions) ─► POST /predict/batch (X-Client: scoring)
```

- **Passer par l'API**, pas par le modèle : le scoring est un vrai client, ses prédictions ont la même version, la
  même trace et les mêmes métriques que celles de l'interface.
- **Une prédiction par match** : un match visible plusieurs pas de suite (horizon de 7 jours) n'est prédit qu'une
  fois, à la première occasion, avec le classement publié pour le tournoi.
- **Pas de modèle en production** (début de démo, toutes les campagnes refusées) : rien à faire, le flow le dit et
  réussit ; ce n'est pas un incident. Une API injoignable, si : le flow échoue (une alerte).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from pydantic import BaseModel

from tennis_mlops.api.client import ApiClient, ApiError
from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig, Settings
from tennis_mlops.flows.ingest import retry_on_outage
from tennis_mlops.ingestion.client import MockSourceClient
from tennis_mlops.serving.persistence import ApiStore
from tennis_mlops.serving.schemas import MatchRequest, PlayerInput

FLOW_NAME = "scoring-atp"
CLIENT_NAME = "scoring"
SERVICE_UNAVAILABLE = 503


class ScoringReport(BaseModel):
    """Bilan d'un passage de scoring."""

    simulated_now: date | None = None
    fixtures: int = 0
    already_predicted: int = 0
    too_early: int = 0  # date pas postérieure à l'historique de l'API (match du jour même, déjà en cours)
    predicted: int = 0
    model_version: str | None = None
    skipped: str | None = None

    def summary(self) -> str:
        """Une ligne pour les logs."""
        if self.skipped:
            return f"scoring sauté : {self.skipped}"
        version = f" par v{self.model_version}" if self.model_version else ""
        return (f"{self.predicted} match(s) prédit(s){version} ; {self.fixtures} à venir, {self.already_predicted} "
                f"déjà prédit(s), {self.too_early} trop proche(s) de l'historique")  # fmt: skip


class ScoringDependencies:
    """Source, API et base ; les tests fournissent les leurs."""

    def __init__(self, source: MockSourceClient, api: ApiClient, store: ApiStore) -> None:
        self.source, self.api, self.store = source, api, store

    def close(self) -> None:
        """Ferme les connexions HTTP."""
        self.source.close()
        self.api.close()


def default_dependencies(config: ProjectConfig, settings: Settings) -> ScoringDependencies:
    """Les services de la stack."""
    return ScoringDependencies(
        MockSourceClient(str(settings.mock_source_url), config.ingestion.http_timeout_s),
        ApiClient(str(settings.api_url), CLIENT_NAME),
        ApiStore.from_url(settings.sqlalchemy_url()),
    )


dependency_factory: Callable[[ProjectConfig, Settings], ScoringDependencies] = default_dependencies


@contextmanager
def _dependencies(config: ProjectConfig) -> Iterator[ScoringDependencies]:
    deps = dependency_factory(config, Settings())
    try:
        yield deps
    finally:
        deps.close()


@task(name="GET /fixtures", retries=3, retry_delay_seconds=[1, 2, 4], retry_condition_fn=retry_on_outage,
      cache_policy=NO_CACHE)  # fmt: skip
def fetch_fixtures(source: MockSourceClient, days: int | None) -> tuple[date, list[dict[str, Any]]]:
    """Les matchs à venir, joueurs connus (classement publié pour le tournoi)."""
    payload = source.fixtures(days)
    return payload.simulated_now, list(payload.data.get("fixtures", []))


@task(name="GET /model", cache_policy=NO_CACHE)
def served_model(api: ApiClient) -> dict[str, Any]:
    """Version servie et dernière journée de l'historique de l'API."""
    return api.model()


@task(name="POST /predict/batch", task_run_name="prédire {count} match(s)", cache_policy=NO_CACHE)
def predict(api: ApiClient, requests: list[MatchRequest], count: int) -> str:
    """Un appel de l'API ; renvoie la version qui a prédit."""
    return api.predict_batch(requests).model.version


def fixture_request(fixture: dict[str, Any]) -> MatchRequest:
    """Un match à venir de la source → requête de l'API (le classement du tournoi remplace le dernier connu)."""

    def player(entry: dict[str, Any]) -> PlayerInput:
        return PlayerInput(
            id=entry["id"], rank=entry.get("rank"), rank_points=entry.get("rank_points"), age=entry.get("age"),
            height_cm=_height(entry.get("ht")), hand=_hand(entry.get("hand")),
        )  # fmt: skip

    return MatchRequest(
        match_id=fixture["match_id"], match_date=fixture["match_date"], surface=fixture.get("surface"),
        tourney_level=fixture.get("tourney_level"), round=fixture.get("round"), best_of=fixture.get("best_of"),
        tourney_name=(fixture.get("tourney_name") or None) and str(fixture["tourney_name"])[:120],
        player1=player(fixture["player1"]), player2=player(fixture["player2"]),
    )  # fmt: skip


@flow(name=FLOW_NAME)
def scoring_flow(days: int | None = None, config_path: str | None = None) -> ScoringReport:
    """Prédit, via l'API, les matchs à venir qui ne l'ont pas encore été."""
    config = ProjectConfig.from_yaml(Path(config_path) if config_path else DEFAULT_CONFIG_PATH)
    logger = get_run_logger()
    report = ScoringReport()
    with _dependencies(config) as deps:
        try:
            info = served_model(deps.api)
        except ApiError as exc:
            if exc.status is None:
                raise
            info = {"served": None}
        if not info.get("served"):
            report.skipped = "aucun modèle en production pour l'instant"
            logger.warning(report.summary())
            return report
        last_day = (info.get("features") or {}).get("last_day")
        report.simulated_now, fixtures = fetch_fixtures(deps.source, days)
        report.fixtures = len(fixtures)
        done = deps.store.predicted_match_ids([f["match_id"] for f in fixtures], client=CLIENT_NAME)
        todo = []
        for fixture in fixtures:
            if fixture["match_id"] in done:
                report.already_predicted += 1
            elif last_day and str(fixture["match_date"]) <= last_day:
                report.too_early += 1
            else:
                todo.append(fixture_request(fixture))
        size = int((info.get("limits") or {}).get("max_batch") or config.api.max_batch)  # la limite de l'API servie
        for chunk in _chunks(todo, size):
            try:
                report.model_version = predict(deps.api, chunk, len(chunk))
            except ApiError as exc:
                if exc.status != SERVICE_UNAVAILABLE:
                    raise
                report.skipped = f"l'API ne sert plus de modèle : {exc.message}"
                break
            report.predicted += len(chunk)
    logger.info(report.summary())
    return report


def _chunks(items: Sequence[MatchRequest], size: int) -> Iterator[list[MatchRequest]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _height(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) and 150 <= value <= 230 else None


def _hand(value: object) -> Any:  # noqa: ANN401
    return value if value in {"R", "L", "U", "A"} else None


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m tennis_mlops.flows.scoring` : un passage, sans planning (`make score`)."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.flows.scoring", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--days", type=int, default=None, help="horizon en jours (défaut : celui de la source)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    print(scoring_flow(args.days, str(args.config)).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
