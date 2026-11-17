"""L'exporteur de la chaîne : sondes de santé, horloge et chaos de la source, runs Prefect."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
from prometheus_client import CollectorRegistry, generate_latest

from tennis_mlops.monitoring.__main__ import targets
from tennis_mlops.monitoring.exporter import ChainCollector, Targets, _timestamp

TARGETS = Targets("http://mlflow", "http://source", "http://prefect/api", "http://api", "http://sink")
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC).timestamp()

FLOWS = [{"id": "f1", "name": "ingestion-atp"}, {"id": "f2", "name": "entrainement-atp"}]
FLOW_RUNS = [  # du plus récent au plus ancien
    {"id": "r3", "flow_id": "f1", "state_name": "Dégradé", "state_type": "COMPLETED", "run_count": 1,
     "end_time": "2026-11-16T11:58:00Z", "total_run_time": 12.5},
    {"id": "r2", "flow_id": "f1", "state_name": "Completed", "state_type": "COMPLETED", "run_count": 2,
     "end_time": "2026-11-16T11:56:00Z", "total_run_time": 8.0},
    {"id": "r1", "flow_id": "f2", "state_name": "Running", "state_type": "RUNNING", "run_count": 1,
     "end_time": None, "total_run_time": 3.0},
]  # fmt: skip
TASK_RUNS = [{"flow_run_id": "r3", "run_count": 3}, {"flow_run_id": "r2", "run_count": 1}]


def _handler(down: set[str], prefect_fails: bool = False) -> httpx.MockTransport:
    seen: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host in down:
            raise httpx.ConnectError("refusé", request=request)
        path = request.url.path
        if path.endswith("/health"):
            body = {"status": "ok"}
            if host == "source":
                body |= {"simulated_now": "2015-03-02", "chaos_mode": "503"}
            return httpx.Response(200, json=body)
        if prefect_fails:
            return httpx.Response(500)
        seen.append(json.loads(request.content))
        data = {"/api/flows/filter": FLOWS, "/api/flow_runs/filter": FLOW_RUNS, "/api/task_runs/filter": TASK_RUNS}
        return httpx.Response(200, json=data[path])

    transport = httpx.MockTransport(handle)
    transport.seen = seen  # type: ignore[attr-defined]
    return transport


def _scrape(collector: ChainCollector) -> str:
    registry = CollectorRegistry()
    registry.register(collector)
    return generate_latest(registry).decode()


def _value(text: str, sample: str) -> float | None:
    """La valeur d'une ligne de métrique (`nom{labels}`), ou None si elle est absente."""
    for line in text.splitlines():
        if not line.startswith("#") and line.rsplit(" ", 1)[0] == sample:
            return float(line.rsplit(" ", 1)[1])
    return None


def test_everything_up() -> None:
    transport = _handler(down=set())
    text = _scrape(ChainCollector(TARGETS, httpx.Client(transport=transport), clock=lambda: NOW))
    for service in ("mlflow", "mock_source", "prefect", "api", "alert_sink"):
        assert f'tennis_service_up{{service="{service}"}} 1.0' in text
    assert _value(text, "tennis_simulated_clock_timestamp") == _timestamp("2015-03-02")
    assert 'tennis_source_chaos{mode="503"} 1.0' in text and 'tennis_source_chaos{mode="off"} 0.0' in text
    assert 'tennis_prefect_flow_runs{flow="ingestion-atp",state="Dégradé"} 1.0' in text
    assert 'tennis_prefect_flow_runs{flow="entrainement-atp",state="Running"} 1.0' in text
    assert 'tennis_prefect_flow_run_seconds{flow="ingestion-atp"} 12.5' in text  # le plus récent terminé
    assert 'tennis_prefect_flow_run_seconds{flow="entrainement-atp"}' not in text  # encore en cours
    assert 'tennis_prefect_retries{flow="ingestion-atp"} 3.0' in text  # 1 relance du flow r2 + 2 de la task de r3
    assert "tennis_exporter_prefect_up 1.0" in text
    body = transport.seen[1]  # type: ignore[attr-defined]
    assert body["flow_runs"]["start_time"]["after_"].startswith("2026-11-15T12:00")  # les dernières 24 h


def test_a_service_down_is_counted_not_fatal() -> None:
    client = httpx.Client(transport=_handler(down={"mlflow", "source"}))
    text = _scrape(ChainCollector(TARGETS, client, clock=lambda: NOW))
    assert 'tennis_service_up{service="mlflow"} 0.0' in text and 'tennis_service_up{service="api"} 1.0' in text
    assert _value(text, 'tennis_source_chaos{mode="off"}') is None
    assert _value(text, "tennis_simulated_clock_timestamp") is None


def test_prefect_errors_are_reported_and_cached() -> None:
    now = [NOW]
    transport = _handler(down=set(), prefect_fails=True)
    collector = ChainCollector(TARGETS, httpx.Client(transport=transport), clock=lambda: now[0], cache_s=10)
    assert "tennis_exporter_prefect_up 0.0" in _scrape(collector)
    assert "tennis_prefect_flow_runs{" not in _scrape(collector)


def test_targets_come_from_the_settings() -> None:
    from tennis_mlops.config import Settings

    t = targets(Settings(prefect_api_url="http://prefect:4200/api/", mlflow_tracking_uri="http://mlflow:7282"))  # type: ignore[arg-type]
    assert t.prefect_api == "http://prefect:4200/api" and t.probes()["prefect"] == "http://prefect:4200/api/health"
    assert t.probes()["mlflow"] == "http://mlflow:7282/health"
