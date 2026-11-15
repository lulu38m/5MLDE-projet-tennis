"""Feedback et qualité en ligne ; état du service : prêt, dégradé (S8), pas prêt ; rechargement ; métriques."""

from datetime import timedelta
from pathlib import Path

import pytest

from tests.support.serving import ServingWorld, match_body, serving_world, train_pipeline
from tests.support.synthetic import synthetic_matches


@pytest.fixture
def world(tmp_path: Path) -> ServingWorld:
    return serving_world(tmp_path, admin_token="secret").ready()


def test_feedback_judges_every_prediction_of_the_match(world: ServingWorld) -> None:
    first = world.client.post("/predict", json=match_body(match_id="m-1")).json()
    favourite = first["predicted_winner_id"]
    underdog = 100_002 if favourite == 100_001 else 100_001

    response = world.client.post("/feedback", json={"match_id": "m-1", "winner_id": favourite},
                                 headers={"X-Client": "feedback"})  # fmt: skip
    data = response.json()
    assert response.status_code == 200 and not data["updated"]
    (outcome,) = data["predictions"]
    assert outcome["correct"] and outcome["model_version"] == "1" and 0 < outcome["log_loss"] < 0.7

    corrected = world.client.post("/feedback", json={"match_id": "m-1", "winner_id": underdog}).json()
    assert corrected["updated"] and not corrected["predictions"][0]["correct"]  # résultat remplacé, pas doublé

    wrong = world.client.post("/feedback", json={"match_id": "m-1", "winner_id": 123})
    assert wrong.status_code == 422 and wrong.json()["error"] == "inconsistent_feedback"

    unknown = world.client.post("/feedback", json={"match_id": "jamais-predit", "winner_id": 100_003}).json()
    assert unknown["predictions"] == []  # gardé quand même : une prédiction peut arriver par une autre voie

    online = world.client.get("/model").json()["online"]
    assert online["predictions"] == 1 and online["with_feedback"] == 1 and online["accuracy"] == 0.0
    world.flush()
    (row,) = world.rows("SELECT actual_winner_id, correct, winner_probability FROM prediction_outcomes")
    assert row["actual_winner_id"] == underdog and row["correct"] == 0 and row["winner_probability"] < 0.5


def test_ready_degraded_and_not_ready(tmp_path: Path) -> None:
    world = serving_world(tmp_path)
    world.registry.down = True
    world.ready()
    not_ready = world.client.get("/ready")
    assert not_ready.status_code == 503 and not_ready.json()["status"] == "not_ready"
    refused = world.client.post("/predict", json=match_body())
    assert refused.status_code == 503 and "aucun modèle servi" in refused.json()["detail"]

    world.registry.down = False
    world.service.model.refresh()
    ready = world.client.get("/ready").json()
    assert (ready["status"], ready["reasons"]) == ("ready", [])
    assert {k: ready["checks"][k] for k in ("model", "registry", "database")} == {
        "model": "1", "registry": "ok", "database": "ok"}  # fmt: skip

    world.registry.down = True  # S8 : MLflow arrêté après le chargement
    world.service.model.refresh()
    degraded = world.client.get("/ready")
    assert degraded.status_code == 200 and degraded.json()["status"] == "degraded"
    assert "v1 servie depuis le cache" in degraded.json()["reasons"][0]
    served = world.client.post("/predict", json=match_body())
    assert served.status_code == 200 and served.headers["X-Model-Version"] == "1"
    assert world.client.get("/model").json()["registry"]["stale"]


def test_admin_reload_serves_a_rollback_at_once(world: ServingWorld) -> None:
    world.registry.add("2", train_pipeline(world.frame, c=0.01))
    world.registry.production = "2"
    assert world.client.post("/admin/reload").status_code == 403
    reloaded = world.client.post("/admin/reload", headers={"X-Admin-Token": "secret"}).json()
    assert reloaded["model"] == {"changed": True, "version": "2", "previous_version": "1", "error": None}
    assert world.client.post("/predict", json=match_body()).headers["X-Model-Version"] == "2"


def test_a_model_trained_on_other_features_is_not_served(tmp_path: Path) -> None:
    world = serving_world(tmp_path)
    world.registry.add("1", world.registry.models["1"], features_version="0")
    world.ready()
    ready = world.client.get("/ready").json()
    assert ready["status"] == "not_ready" and "réentraîner" in ready["reasons"][0]


def test_new_matches_are_replayed(world: ServingWorld) -> None:
    before = world.service.features.current
    assert world.service.features.refresh().changed is False
    start = world.last_day + timedelta(days=7)
    later = synthetic_matches(start=start, weeks=2, matches_per_week=30, players=40, seed=11)
    later["match_id"] = "nouveau-" + later["match_id"]
    world.add_matches(later)
    refresh = world.service.features.refresh()
    assert refresh.changed and refresh.snapshot.matches == before.matches + len(later)  # type: ignore[union-attr]
    assert refresh.snapshot.last_day > before.last_day  # type: ignore[union-attr,operator]


def test_metrics(world: ServingWorld) -> None:
    world.client.post("/predict", json=match_body(), headers={"X-Client": "locust"})
    world.client.post("/predict", json=match_body(surface="Terre"))
    text = world.client.get("/metrics").text
    assert 'tennis_api_requests_total{method="POST",path="/predict",status="200"} 1.0' in text
    assert 'tennis_api_requests_total{method="POST",path="/predict",status="422"} 1.0' in text
    assert 'tennis_api_predictions_total{client="locust",model_version="1"} 1.0' in text
    assert 'tennis_api_model_info{name="atp-winner",version="1"} 1.0' in text
    assert 'tennis_api_errors_total{type="validation"} 1.0' in text
    assert "tennis_api_ready 2.0" in text and "tennis_api_request_duration_seconds_bucket" in text
    assert world.client.get("/health").json() == {"status": "ok"}
