"""Endpoints de consultation de l'interface : joueurs, statistiques, face-à-face, matchs prédits à l'avance."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from tests.support.serving import ServingWorld, match_body, serving_world


@pytest.fixture
def world(tmp_path: Path) -> ServingWorld:
    return serving_world(tmp_path).ready()


def test_players(world: ServingWorld) -> None:
    data = world.client.get("/players", params={"search": "joueur 3", "limit": 5}).json()
    assert data["query"] == "joueur 3" and data["as_of"] == world.last_day.isoformat()
    assert 0 < data["count"] <= 5 and all("3" in p["name"] for p in data["players"])
    assert world.client.get("/players", params={"limit": 0}).status_code == 422


def test_player_stats(world: ServingWorld) -> None:
    response = world.client.get("/players/100001/stats", headers={"X-Client": "ui"})
    data = response.json()
    assert response.status_code == 200 and data["player"]["name"] == "Joueur 1" and data["ratings"]["elo"] > 0
    later = (world.last_day + timedelta(days=30)).isoformat()
    assert (
        world.client.get("/players/100001/stats", params={"as_of": later}).json()["as_of"] == world.last_day.isoformat()
    )
    missing = world.client.get("/players/999999/stats")
    assert missing.status_code == 404 and missing.json()["error"] == "unknown_player"
    world.flush()
    assert world.rows("SELECT path, client FROM api_requests WHERE status_code = 200")[0] == {
        "path": "/players/{player_id}/stats", "client": "ui"}  # fmt: skip
    assert world.rows("SELECT error_type FROM errors") == [{"error_type": "unknown_player"}]


def test_h2h(world: ServingWorld) -> None:
    data = world.client.get("/h2h", params={"p1": 100001, "p2": 100002}).json()
    assert data["player1"]["name"] == "Joueur 1" and data["total"] == data["p1_wins"] + data["p2_wins"]
    same = world.client.get("/h2h", params={"p1": 100001, "p2": 100001})
    assert same.status_code == 422 and same.json()["error"] == "invalid_match"
    assert world.client.get("/h2h", params={"p1": 100001}).status_code == 422


def test_fixture_predictions(world: ServingWorld) -> None:
    body = match_body(match_id="2099-7-1", tourney_name="Open de demain")
    world.client.post("/predict/batch", json={"matches": [body]}, headers={"X-Client": "scoring"})
    world.flush()
    data = world.client.get("/fixtures/predictions", params={"days": 7}).json()
    assert data["count"] == 1 and data["client"] == "scoring" and data["played"] == 0 and data["accuracy"] is None
    (fixture,) = data["fixtures"]
    assert (fixture["tourney_name"], fixture["status"], fixture["model_version"]) == ("Open de demain", "upcoming", "1")
    assert world.client.get("/fixtures/predictions", params={"client": "ui"}).json()["count"] == 0


def test_consultation_without_history_is_unavailable(tmp_path: Path) -> None:
    world = serving_world(tmp_path)  # rien n'est encore chargé
    for path in ("/players", "/players/100001/stats", "/h2h?p1=100001&p2=100002", "/fixtures/predictions"):
        response = world.client.get(path)
        assert response.status_code == 503 and response.json()["error"] == "unavailable", path


def test_consultation_works_without_a_model(tmp_path: Path) -> None:
    """Pas de version en production : on ne prédit pas, mais on peut toujours explorer les joueurs."""
    world = serving_world(tmp_path)
    world.registry.production = None
    world.ready()
    assert world.client.post("/predict", json=match_body()).status_code == 503
    assert world.client.get("/players/100001/stats").status_code == 200
