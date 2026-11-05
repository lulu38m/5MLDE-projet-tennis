from datetime import date

import pytest

from .conftest import MockEnv

DATA_ENDPOINTS = ["/results?since=2019-01-01", "/fixtures", "/tournaments", "/rankings"]


def test_every_response_carries_simulation_headers(env: MockEnv) -> None:
    client = env.at(date(2019, 1, 8))
    for path in ["/health", "/admin/clock", "/tournaments/nope/results", *DATA_ENDPOINTS]:
        response = client.get(path)
        assert response.headers["X-Simulated-Now"] == "2019-01-08", path
        assert response.headers["X-Dataset-Version"] == env.lock.dataset_version, path


def test_health(env: MockEnv) -> None:
    body = env.client.get("/health").json()
    assert body == {
        "status": "ok",
        "dataset_version": env.lock.dataset_version,
        "simulated_now": "2019-01-01",
        "matches_loaded": 14,
        "chaos_mode": "off",
    }


def test_results_endpoint(env: MockEnv) -> None:
    client = env.at(date(2019, 1, 11))
    body = client.get("/results", params={"since": "2019-01-01"}).json()
    assert (body["until"], body["count"], body["reason"]) == ("2019-01-11", 6, None)
    assert set(body["results"][0]) >= {"match_id", "winner", "loser", "score", "minutes", "status"}
    assert client.get("/results").status_code == 422  # since obligatoire
    assert client.get("/results", params={"since": "2019-01-01", "until": "2019-01-12"}).status_code == 400
    assert client.get("/results", params={"since": "2019-01-10", "until": "2019-01-09"}).status_code == 400


def test_empty_period_returns_200_with_reason(env: MockEnv) -> None:
    body = env.at(date(2019, 6, 1)).get("/results", params={"since": "2019-05-01"}).json()
    assert (body["count"], body["reason"]) == (0, "Pause de test")
    fixtures = env.client.get("/fixtures").json()
    assert fixtures["count"] == 0 and fixtures["reason"]


def test_fixtures_endpoint_has_no_result_fields(env: MockEnv) -> None:
    body = env.at(date(2019, 1, 8)).get("/fixtures").json()
    assert body["horizon_days"] == 7 and body["count"] == 2
    for fixture in body["fixtures"]:
        assert {"winner", "loser", "score", "minutes", "status"}.isdisjoint(fixture)
        assert {"player1", "player2"} <= set(fixture)
    assert env.client.get("/fixtures", params={"days": 31}).status_code == 422
    assert env.client.get("/fixtures", params={"days": 1}).json()["count"] == 0


def test_tournament_endpoints(env: MockEnv) -> None:
    client = env.at(date(2019, 1, 11))
    lot = client.get("/tournaments/2019-0001/results").json()
    assert (lot["expected_matches"], lot["count"], lot["tournament"]["status"]) == (7, 6, "in_progress")
    assert client.get("/tournaments/2020-0002/results").status_code == 404  # pas encore annoncé
    listing = client.get("/tournaments", params={"status": "in_progress"}).json()
    assert [t["tourney_id"] for t in listing["tournaments"]] == ["2019-0001"]


def test_rankings_endpoint(env: MockEnv) -> None:
    client = env.at(date(2019, 1, 20))
    body = client.get("/rankings", params={"limit": 2}).json()
    assert (body["requested_date"], body["ranking_date"], body["count"]) == ("2019-01-20", "2019-01-14", 2)
    assert client.get("/rankings", params={"date": "2019-01-21"}).status_code == 400
    early = env.at(date(2019, 1, 2)).get("/rankings").json()
    assert early["ranking_date"] is None and early["reason"]


def test_player_history_endpoint(env: MockEnv) -> None:
    client = env.at(date(2019, 1, 20))
    assert client.get("/players/1/history").status_code == 422  # as_of obligatoire
    assert client.get("/players/1/history", params={"as_of": "2019-01-21"}).status_code == 400
    assert client.get("/players/10/history", params={"as_of": "2019-01-20"}).status_code == 404
    assert client.get("/players/9/history", params={"as_of": "2019-01-20"}).json()["is_debutant"] is True
    body = client.get("/players/1/history", params={"as_of": "2019-01-20", "limit": 1}).json()
    assert (body["wins"], len(body["matches"]), body["matches"][0]["round"]) == (3, 1, "F")


def test_clock_administration(env: MockEnv) -> None:
    client = env.client
    assert client.post("/admin/clock", json={}).status_code == 422
    assert client.post("/admin/clock", json={"date": "2030-01-01"}).status_code == 422
    assert client.post("/admin/clock", json={"date": "2019-01-05", "speed": 0}).json()["now"] == "2019-01-05"
    advanced = client.post("/admin/clock/advance", json={"days": 9}).json()
    assert advanced == {
        "previous": "2019-01-05",
        "now": "2019-01-14",
        "newly_completed_matches": 7,
        "newly_completed_tournaments": ["2019-0001"],
    }
    assert client.post("/admin/clock/advance", json={"days": 1000}).status_code == 422
    assert client.get("/admin/clock").json()["now"] == "2019-01-14"


def test_chaos_503_then_recovery(env: MockEnv) -> None:
    client = env.at(date(2019, 1, 11))
    client.post("/admin/chaos", json={"mode": "503", "failures": 2})
    assert client.get("/admin/clock").status_code == 200  # l'administration n'est jamais affectée
    first, second, third = (client.get("/results", params={"since": "2019-01-01"}) for _ in range(3))
    assert (first.status_code, second.status_code, third.status_code) == (503, 503, 200)
    assert first.headers["Retry-After"] == "1" and first.headers["X-Chaos-Mode"] == "503"
    assert client.get("/admin/chaos").json()["mode"] == "off"


def test_chaos_slow(env: MockEnv) -> None:
    client = env.at(date(2019, 1, 11))
    client.post("/admin/chaos", json={"mode": "slow", "delay_s": 0.01, "failures": 1})
    response = client.get("/tournaments")
    assert response.status_code == 200 and response.headers["X-Chaos-Mode"] == "slow"


def test_chaos_corrupt_and_schema(env: MockEnv) -> None:
    client = env.at(date(2019, 1, 14))
    clean = client.get("/tournaments/2019-0001/results").json()
    client.post("/admin/chaos", json={"mode": "corrupt"})
    corrupted = client.get("/tournaments/2019-0001/results").json()
    assert corrupted != clean
    client.post("/admin/chaos", json={"mode": "schema"})
    broken = client.get("/results", params={"since": "2019-01-01"}).json()["results"][0]
    assert "matchId" in broken and "score" not in broken
    client.post("/admin/chaos", json={"mode": "off"})
    assert client.get("/tournaments/2019-0001/results").json() == clean


@pytest.mark.parametrize("body", [{"mode": "explode"}, {"mode": "503", "failures": 0}, {"mode": "slow", "delay_s": -1}])
def test_invalid_chaos_settings(env: MockEnv, body: dict[str, object]) -> None:
    assert env.client.post("/admin/chaos", json=body).status_code == 422
