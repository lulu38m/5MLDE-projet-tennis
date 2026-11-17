"""CPU et mémoire des conteneurs lus sur l'API Docker (simulée)."""

from __future__ import annotations

import json
from typing import Any

import httpx
from prometheus_client import CollectorRegistry, generate_latest

from tennis_mlops.monitoring.containers import ContainerCollector, parse_stats

GIB = 1_073_741_824
V2 = {"cpu_stats": {"cpu_usage": {"total_usage": 2_500_000_000}},
      "memory_stats": {"usage": 300_000_000, "limit": GIB, "stats": {"inactive_file": 100_000_000}}}  # fmt: skip
V1 = {"cpu_stats": {"cpu_usage": {"total_usage": 1_000_000_000}},
      "memory_stats": {"usage": 50_000_000, "limit": GIB // 2, "stats": {"total_inactive_file": 10**7}}}  # fmt: skip


def docker(stats: dict[str, dict[str, Any]], fail: bool = False) -> tuple[httpx.Client, list[str]]:
    """Une API Docker : deux conteneurs du projet, dont un qui s'arrête entre la liste et la lecture."""
    asked: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("pas de socket", request=request)
        asked.append(str(request.url))
        if request.url.path == "/containers/json":
            filters = json.loads(request.url.params["filters"])
            assert filters == {"label": ["com.docker.compose.project=tennis-mlops"], "status": ["running"]}
            listed = [{"Id": cid, "Names": [f"/tennis-mlops-{cid}-1"], "Labels": {"com.docker.compose.service": cid}}
                      for cid in ("api", "postgres", "gone")]  # fmt: skip
            return httpx.Response(200, json=listed)
        cid = request.url.path.split("/")[2]
        assert request.url.params["one-shot"] == "true"
        return httpx.Response(200, json=stats[cid]) if cid in stats else httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handle), base_url="http://docker"), asked


def test_parse_stats_matches_docker_stats() -> None:
    v2 = parse_stats("api", "c", V2)
    assert v2 is not None
    assert v2.cpu_seconds == 2.5
    assert v2.memory_bytes == 200_000_000  # usage moins le cache de fichiers inactif
    assert v2.memory_limit_bytes == GIB
    v1 = parse_stats("db", "c", V1)
    assert v1 is not None
    assert v1.memory_bytes == 40_000_000
    assert parse_stats("x", "c", {"cpu_stats": {}, "memory_stats": {}}) is None  # conteneur arrêté


def test_collector_exposes_project_containers_only() -> None:
    http, asked = docker({"api": V2, "postgres": V1})
    registry = CollectorRegistry()
    registry.register(ContainerCollector("tennis-mlops", http))
    text = generate_latest(registry).decode()
    assert "tennis_docker_up 1.0" in text
    assert 'tennis_container_cpu_seconds_total{container="tennis-mlops-api-1",service="api"} 2.5' in text
    assert 'tennis_container_memory_bytes{container="tennis-mlops-postgres-1",service="postgres"} 4e+07' in text
    assert 'tennis_container_memory_limit_bytes{container="tennis-mlops-api-1",service="api"}' in text
    assert "gone" not in text  # arrêté entre la liste et la lecture : ignoré
    assert sum("/stats" in url for url in asked) == 3


def test_unreachable_docker_says_so() -> None:
    http, _ = docker({}, fail=True)
    registry = CollectorRegistry()
    registry.register(ContainerCollector("tennis-mlops", http))
    text = generate_latest(registry).decode()
    assert "tennis_docker_up 0.0" in text
    assert "tennis_container_cpu_seconds_total{" not in text
