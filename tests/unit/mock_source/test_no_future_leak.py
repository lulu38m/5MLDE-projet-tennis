"""Garantie centrale du simulateur : quelle que soit la date simulée, aucune réponse ne révèle le futur.

Le test balaie chaque jour du mini-dataset ; `test_real_data_never_leaks` rejoue la même vérification sur les
vraies données (si `make data` a été lancé), à 30 dates réparties sur toute la fenêtre de rejeu.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig
from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.errors import DataIntegrityError
from tennis_mlops.data.lockfile import DataLock
from tests.support.mock_world import MockEnv


def assert_no_leak(client: TestClient, now: date) -> None:
    today = now.isoformat()
    results = client.get("/results", params={"since": "2000-01-01"}).json()["results"]
    assert all(r["match_date"] < today for r in results), "résultat du futur"
    played = {r["match_id"] for r in results}
    losers = {(r["tourney_id"], r["loser"]["id"]) for r in results if r["round"] != "RR"}

    for fixture in client.get("/fixtures", params={"days": 30}).json()["fixtures"]:
        assert fixture["match_date"] >= today, "match à venir déjà joué"
        assert fixture["match_id"] not in played
        for player in (fixture["player1"], fixture["player2"]):
            assert (fixture["tourney_id"], player["id"]) not in losers, "joueur déjà éliminé annoncé au tour suivant"

    for tournament in client.get("/tournaments").json()["tournaments"]:
        assert tournament["start_date"] <= (now + timedelta(days=14)).isoformat()
        lot = client.get(f"/tournaments/{tournament['tourney_id']}/results").json()
        assert all(r["match_date"] < today for r in lot["results"])
        assert lot["count"] == tournament["completed_matches"]

    ranking_date = client.get("/rankings").json()["ranking_date"]
    assert ranking_date is None or ranking_date <= today


TOURNAMENT_WINDOWS = [(date(2018, 12, 20), date(2019, 1, 20)), (date(2019, 2, 25), date(2019, 3, 6)),
                      (date(2019, 12, 20), date(2020, 1, 20))]  # fmt: skip


def sweep_days() -> list[date]:
    """Chaque jour autour des tournois, une fois par semaine ailleurs."""
    days = {d + timedelta(days=i) for d, end in TOURNAMENT_WINDOWS for i in range((end - d).days + 1)}
    day = date(2018, 12, 20)
    while day <= date(2020, 1, 20):
        days.add(day)
        day += timedelta(days=7)
    return sorted(days)


def test_synthetic_data_never_leaks(env: MockEnv) -> None:
    for day in sweep_days():
        assert_no_leak(env.at(day), day)


@pytest.mark.data
def test_real_data_never_leaks() -> None:
    config = ProjectConfig.from_yaml(DEFAULT_CONFIG_PATH)
    root = DEFAULT_CONFIG_PATH.parents[1]
    raw = config.data.resolved_raw_dir(root)
    try:
        DataLock.load(config.data.resolved_lock_file(root)).verify_directory(
            raw, DatasetCatalog.from_config(config.data).names()
        )
    except (DataIntegrityError, FileNotFoundError):
        pytest.skip("données réelles absentes : lancer `make data`")

    from mock_source.service import MockSettings, build_app

    with TestClient(build_app(MockSettings(config_path=DEFAULT_CONFIG_PATH))) as client:
        start, end = config.simulation.replay_start, config.simulation.replay_end
        step = (end - start) / 29
        for i in range(30):
            now = start + step * i
            client.post("/admin/clock", json={"date": now.isoformat()})
            assert_no_leak_light(client, now)


def assert_no_leak_light(client: TestClient, now: date) -> None:
    """Version allégée pour les vraies données (sans relire chaque tournoi)."""
    today = now.isoformat()
    since = (now - timedelta(days=60)).isoformat()
    results = client.get("/results", params={"since": since}).json()["results"]
    assert all(r["match_date"] < today for r in results)
    eliminated = {(r["tourney_id"], r["loser"]["id"]) for r in results if r["round"] != "RR"}
    for fixture in client.get("/fixtures", params={"days": 14}).json()["fixtures"]:
        assert fixture["match_date"] >= today
        for player in (fixture["player1"], fixture["player2"]):
            assert (fixture["tourney_id"], player["id"]) not in eliminated
    ranking_date = client.get("/rankings").json()["ranking_date"]
    assert ranking_date is None or ranking_date <= today
