from datetime import date

import httpx
import pytest

from mock_source.chaos import ChaosMode, ChaosSettings
from tennis_mlops.data.errors import SourceRequestError, SourceUnavailableError
from tennis_mlops.ingestion.client import MockSourceClient
from tests.support.mock_world import MockEnv


def client_for(env: MockEnv) -> MockSourceClient:
    return MockSourceClient("", 5, http=env.client)


def fake_client(handler: httpx.MockTransport) -> MockSourceClient:
    return MockSourceClient("", 5, http=httpx.Client(transport=handler, base_url="http://source"))


def test_endpoints_return_payload_with_source_context(env: MockEnv) -> None:
    env.at(date(2019, 1, 14))
    client = client_for(env)
    clock = client.clock()
    assert (clock.endpoint, clock.simulated_now, clock.dataset_version) == (
        "/admin/clock",
        date(2019, 1, 14),
        env.lock.dataset_version,
    )
    listing = client.tournaments(date(2019, 1, 1), date(2019, 1, 14)).data
    assert [t["tourney_id"] for t in listing["tournaments"]] == ["2019-0001"]
    lot = client.tournament_results("2019-0001").data
    assert (lot["expected_matches"], lot["count"]) == (7, 7)
    assert client.results(date(2019, 1, 1), date(2019, 1, 14)).data["count"] == 7


def test_outage_is_retryable_but_refusal_is_not(env: MockEnv) -> None:
    client = client_for(env)
    env.chaos.configure(ChaosSettings(mode=ChaosMode.UNAVAILABLE))
    with pytest.raises(SourceUnavailableError, match="HTTP 503"):
        client.tournament_results("2019-0001")
    env.chaos.configure(ChaosSettings(mode=ChaosMode.OFF))
    with pytest.raises(SourceRequestError, match="HTTP 404"):
        client.tournament_results("inconnu")


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("lent"), httpx.ConnectError("refusé")],
)
def test_network_errors_become_unavailable(error: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(SourceUnavailableError, match="admin/clock"):
        fake_client(httpx.MockTransport(handler)).clock()


def test_unreadable_body_and_missing_header() -> None:
    with pytest.raises(SourceUnavailableError, match="illisible"):
        fake_client(httpx.MockTransport(lambda r: httpx.Response(200, text="<html>"))).clock()
    with pytest.raises(SourceRequestError, match="X-Simulated-Now"):
        fake_client(httpx.MockTransport(lambda r: httpx.Response(200, json={}))).clock()


def test_context_manager_closes_only_its_own_connection(env: MockEnv) -> None:
    with MockSourceClient("http://127.0.0.1:1", 1) as owned:
        assert owned is not None
    with client_for(env) as borrowed:
        borrowed.clock()
    assert env.client.get("/health").status_code == 200  # le client injecté reste ouvert
