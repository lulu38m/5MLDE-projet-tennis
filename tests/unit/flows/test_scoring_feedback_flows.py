"""Scoring et feedback de bout en bout : la source annonce les matchs, l'API les prédit, l'ingestion valide les
résultats, le feedback les remonte et chaque prédiction est jugée."""

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tennis_mlops.api.app import create_app
from tennis_mlops.api.client import ApiClient
from tennis_mlops.config import ApiConfig, ProjectConfig, Settings
from tennis_mlops.features import FeatureConfig
from tennis_mlops.flows import feedback, ingest, scoring
from tennis_mlops.ingestion.client import MockSourceClient
from tennis_mlops.registry.loader import AliasModelLoader
from tennis_mlops.serving.features import FeatureStore
from tennis_mlops.serving.model import ModelHolder
from tennis_mlops.serving.persistence import ApiStore, AsyncWriter
from tennis_mlops.serving.service import PredictionService
from tests.support.serving import FakeRegistry, train_pipeline
from tests.support.synthetic import synthetic_matches
from tests.unit.flows.conftest import FlowWorld


class ApiWorld:
    """Une API branchée sur la base du monde simulé (même fichier SQLite que l'ingestion)."""

    def __init__(self, world: FlowWorld, monkeypatch: pytest.MonkeyPatch, with_model: bool = True) -> None:
        self.registry = FakeRegistry()
        if with_model:
            self.registry.add("1", train_pipeline(synthetic_matches(weeks=30, matches_per_week=20, players=30)))
            self.registry.production = "1"
        world.store.create_schema()
        self.store = ApiStore(world.store.engine)
        self.service = PredictionService(
            features=FeatureStore(world.store, FeatureConfig()),
            model=ModelHolder(AliasModelLoader(self.registry, load=self.registry.load)),  # type: ignore[arg-type]
            store=self.store,
            writer=AsyncWriter(self.store),
            config=ApiConfig(max_batch=2),  # plusieurs appels pour 4 matchs
        )
        self.store.create_schema()
        self.http = TestClient(create_app(self.service, manage_service=False))

        def scoring_deps(*_: object) -> scoring.ScoringDependencies:
            return scoring.ScoringDependencies(MockSourceClient("", 5, http=world.env.client),
                                               ApiClient("", "scoring", http=self.http), self.store)  # fmt: skip

        def feedback_deps(*_: object) -> feedback.FeedbackDependencies:
            return feedback.FeedbackDependencies(ApiClient("", "feedback", http=self.http), self.store)

        monkeypatch.setattr(scoring, "dependency_factory", scoring_deps)
        monkeypatch.setattr(feedback, "dependency_factory", feedback_deps)
        monkeypatch.setattr(ingest, "training_event_emitter", lambda **_: None)

    def refresh(self) -> None:
        self.service.features.refresh()
        self.service.model.refresh()
        self.service.writer.flush()


def test_fixtures_are_scored_once_then_judged_by_validated_results(
    world: FlowWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = ApiWorld(world, monkeypatch)
    world.env.at(date(2019, 2, 1))
    ingest.ingest_flow(str(world.config_path))  # amorçage : l'Alpha Open de janvier en base
    api.refresh()

    world.env.at(date(2019, 2, 28))  # tableau de la Coupe Davis publié : 4 rencontres à venir
    first = scoring.scoring_flow(config_path=str(world.config_path))
    assert (first.fixtures, first.predicted, first.model_version) == (4, 4, "1")
    api.service.writer.flush()  # en vrai, l'écrivain de l'API vide sa file toutes les 0,5 s
    again = scoring.scoring_flow(config_path=str(world.config_path))
    assert (again.predicted, again.already_predicted) == (0, 4)  # un match n'est prédit qu'une fois
    assert {r for r in api.store.predicted_match_ids([f"2019-D001-{n}" for n in (1, 2, 4, 5)], client="scoring")} == {
        "2019-D001-1", "2019-D001-2", "2019-D001-4", "2019-D001-5"}  # fmt: skip

    assert feedback.feedback_flow().awaiting == 0  # résultats pas encore validés : rien à remonter
    world.env.at(date(2019, 3, 10))
    ingest.ingest_flow(str(world.config_path))  # lot de la Coupe Davis validé
    report = feedback.feedback_flow()
    assert (report.sent, report.judged, report.rejected, report.by_version) == (4, 4, 0, {"1": 4})
    assert feedback.feedback_flow().awaiting == 0  # idempotent
    (online,) = api.store.online_metrics()
    assert online.with_feedback == 4 and online.accuracy is not None


def test_no_model_in_production_is_not_an_incident(world: FlowWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    ApiWorld(world, monkeypatch, with_model=False).refresh()
    world.env.at(date(2019, 2, 28))
    state = scoring.scoring_flow(config_path=str(world.config_path), return_state=True)
    assert state.is_completed() and state.result().skipped == "aucun modèle en production pour l'instant"


def test_unreachable_api_fails_the_flow(world: FlowWorld, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def deps(config: ProjectConfig, settings: Settings) -> scoring.ScoringDependencies:
        store = ApiStore.from_url(f"sqlite:///{tmp_path / 'x.sqlite'}")
        api = ApiClient("http://127.0.0.1:9", "scoring", timeout_s=0.5)
        return scoring.ScoringDependencies(MockSourceClient("", 5, http=world.env.client), api, store)

    monkeypatch.setattr(scoring, "dependency_factory", deps)
    state = scoring.scoring_flow(config_path=str(world.config_path), return_state=True)
    assert state.is_failed() and "injoignable" in str(state.result(raise_on_failure=False))


def test_fixture_request_keeps_the_tournament_ranking() -> None:
    fixture = {"match_id": "2019-D001-1", "match_date": "2019-03-01", "surface": "Hard", "tourney_level": "D",
               "round": "RR", "best_of": 3, "player1": {"id": 1, "rank": 1, "rank_points": 999, "ht": 3, "hand": "R"},
               "player2": {"id": 5, "rank": None, "hand": "?"}}  # fmt: skip
    request = scoring.fixture_request(fixture)
    assert request.player1.rank == 1 and request.player1.height_cm is None  # « 3 cm » de la source : ignoré
    assert request.player2.hand is None and request.match_date == date(2019, 3, 1)
