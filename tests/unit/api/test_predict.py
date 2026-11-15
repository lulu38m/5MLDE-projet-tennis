"""`POST /predict` et `/predict/batch` : probabilités, version, traces en base, refus explicites."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from tests.support.serving import ServingWorld, match_body, serving_world


@pytest.fixture
def world(tmp_path: Path) -> ServingWorld:
    return serving_world(tmp_path).ready()


def test_a_prediction_carries_its_version_and_is_traced(world: ServingWorld) -> None:
    response = world.client.post("/predict", json=match_body(match_id="2011-0099-1"),
                                 headers={"X-Request-ID": "essai-1", "X-Client": "ui"})  # fmt: skip
    assert response.status_code == 200
    assert response.headers["X-Model-Version"] == "1" and response.headers["X-Request-ID"] == "essai-1"
    body = response.json()
    p1, p2 = body["player1"]["win_probability"], body["player2"]["win_probability"]
    assert p1 + p2 == pytest.approx(1) and 0 < p1 < 1
    assert body["predicted_winner_id"] == (100_001 if p1 >= 0.5 else 100_002)
    assert body["model"] == {"name": "atp-winner", "version": "1", "run_id": "run-1", "alias": "production"}
    assert body["player1"]["name"] == "Joueur 1" and body["player1"]["known"] and body["warnings"] == []
    last_day = world.last_day
    assert body["features_as_of"] == last_day.isoformat()
    assert body["match_date"] == (last_day + timedelta(days=1)).isoformat()  # pas de date : le lendemain
    assert set(body["features"]) >= {"elo_diff", "surface_elo_diff", "h2h_diff", "surface"}

    world.flush()
    (row,) = world.rows("SELECT * FROM prediction_outcomes")
    assert (row["model_version"], row["client"], row["match_id"]) == ("1", "ui", "2011-0099-1")
    assert row["duration_ms"] > 0 and row["actual_winner_id"] is None  # version, durée… résultat à venir
    (request,) = world.rows("SELECT * FROM api_requests")
    assert (request["request_id"], request["path"], request["status_code"], request["predictions"]) == (
        "essai-1", "/predict", 200, 1)  # fmt: skip
    assert json.loads(world.rows("SELECT features FROM predictions")[0]["features"])["surface"] == "Clay"


def test_overrides_and_unknown_players(world: ServingWorld) -> None:
    body = match_body(player1={"id": 100_001, "rank": 1, "rank_points": 12_000}, player2={"id": 999_999})
    data = world.client.post("/predict", json=body).json()
    assert data["player1"]["rank"] == 1.0 and not data["player2"]["known"]
    assert data["warnings"] == ["player2 (999999) : aucun match dans l'historique, Elo de départ",
                                "player2 (999999) : classement inconnu"]  # fmt: skip
    assert world.service.features.current.builder.players == 40  # type: ignore[union-attr]  # état non modifié


def test_batch(world: ServingWorld) -> None:
    matches = [match_body(100_000 + i, 100_010 + i, match_id=f"m-{i}") for i in range(5)]
    response = world.client.post("/predict/batch", json={"matches": matches}, headers={"X-Client": "scoring"})
    data = response.json()
    assert response.status_code == 200 and data["count"] == 5 and response.headers["X-Model-Version"] == "1"
    assert [p["match_id"] for p in data["predictions"]] == [f"m-{i}" for i in range(5)]
    world.flush()
    assert {r["batch_size"] for r in world.rows("SELECT batch_size FROM predictions")} == {5}
    assert world.rows("SELECT predictions FROM api_requests")[0]["predictions"] == 5


def test_invalid_requests_are_explained_and_traced(world: ServingWorld) -> None:
    same = world.client.post("/predict", json=match_body(100_001, 100_001), headers={"X-Request-ID": "r-422"})
    assert same.status_code == 422 and same.json()["error"] == "validation"
    assert "deux joueurs différents" in same.json()["detail"] and same.headers["X-Request-ID"] == "r-422"

    typo = world.client.post("/predict", json=match_body(surface="Terre"))
    assert typo.status_code == 422 and typo.json()["errors"][0]["loc"] == ["body", "surface"]

    last_day = world.last_day
    past = world.client.post("/predict/batch", json={"matches": [
        match_body(), match_body(match_date=last_day.isoformat())]})  # fmt: skip
    assert past.status_code == 422 and past.json()["error"] == "invalid_match"
    assert past.json()["errors"] == {"index": 1} and "postérieurs" in past.json()["detail"]

    too_many = world.client.post("/predict/batch", json={"matches": [match_body()] * 51})
    assert too_many.status_code == 422 and "50 au plus" in too_many.json()["detail"]

    world.flush()
    errors = world.rows("SELECT request_id, status_code, error_type, message FROM errors ORDER BY id")
    assert [(e["status_code"], e["error_type"]) for e in errors] == [
        (422, "validation"), (422, "validation"), (422, "invalid_match"), (422, "invalid_match")]  # fmt: skip
    assert errors[0]["request_id"] == "r-422"
    assert world.rows("SELECT COUNT(*) AS n FROM predictions")[0]["n"] == 0
    statuses = [r["status_code"] for r in world.rows("SELECT status_code FROM api_requests")]
    assert statuses == [422, 422, 422, 422]


def test_internal_errors_are_traced(world: ServingWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: object) -> None:
        msg = "colonne manquante"
        raise KeyError(msg)

    monkeypatch.setattr(world.service.model.served.model, "predict_proba", broken)  # type: ignore[union-attr]
    client = world.client.__class__(world.app, raise_server_exceptions=False)
    response = client.post("/predict", json=match_body(), headers={"X-Request-ID": "boum"})
    assert response.status_code == 500 and response.json()["error"] == "internal"
    world.flush()
    (error,) = world.rows("SELECT * FROM errors")
    assert error["request_id"] == "boum" and error["message"].startswith("KeyError")
    assert "predict" in json.loads(error["detail"])["traceback"][-1] or json.loads(error["detail"])["traceback"]
    assert world.rows("SELECT status_code FROM api_requests")[0]["status_code"] == 500
