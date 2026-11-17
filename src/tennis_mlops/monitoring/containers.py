"""CPU et mémoire de chaque conteneur de la stack, lus sur l'API Docker (le socket `/var/run/docker.sock`).

C'est ce que fait `docker stats`. On le fait nous-mêmes plutôt qu'avec cAdvisor : depuis Docker 29 et le magasin
d'images containerd (Docker Desktop récent), cAdvisor doit joindre le socket de containerd, qui n'est pas exposé
aux conteneurs sur macOS ; il ne voyait alors aucun conteneur. L'API Docker, elle, répond partout de la même façon.

Seuls les conteneurs du projet compose (`com.docker.compose.project`) sont lus. Les statistiques de tous les
conteneurs sont demandées en parallèle, une fois par collecte (`one-shot` : pas d'attente d'un second échantillon).

- `tennis_container_cpu_seconds_total{service,container}` : temps CPU cumulé (compteur : `rate()` donne des cœurs) ;
- `tennis_container_memory_bytes{service,container}` : mémoire de travail (usage moins le cache de fichiers
  inactif, comme `docker stats`) ;
- `tennis_container_memory_limit_bytes{service,container}` : limite (`deploy.resources.limits.memory`) ;
- `tennis_docker_up` : 0 si le socket ne répond pas.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

SOCKET = "/var/run/docker.sock"
PROJECT_LABEL = "com.docker.compose.project"
SERVICE_LABEL = "com.docker.compose.service"


@dataclass(frozen=True)
class ContainerStats:
    """Une mesure d'un conteneur."""

    service: str
    container: str
    cpu_seconds: float
    memory_bytes: float
    memory_limit_bytes: float | None


def parse_stats(service: str, container: str, stats: dict[str, Any]) -> ContainerStats | None:
    """Une réponse de `GET /containers/{id}/stats` ; None pour un conteneur arrêté (statistiques vides)."""
    cpu = (stats.get("cpu_stats") or {}).get("cpu_usage") or {}
    memory = stats.get("memory_stats") or {}
    if "total_usage" not in cpu or "usage" not in memory:
        return None
    details = memory.get("stats") or {}
    # cgroup v2 : inactive_file ; cgroup v1 : total_inactive_file (même calcul que `docker stats`).
    inactive = details.get("inactive_file", details.get("total_inactive_file", 0))
    used = max(float(memory["usage"]) - float(inactive), 0.0)
    limit = memory.get("limit")
    # Sans `deploy.resources.limits.memory`, Docker renvoie la mémoire de la VM (tous nos services ont une limite).
    return ContainerStats(service, container, float(cpu["total_usage"]) / 1e9, used, float(limit) if limit else None)


class ContainerCollector(Collector):
    """Collecteur Prometheus : une passe sur l'API Docker à chaque lecture."""

    def __init__(self, project: str, http: httpx.Client | None = None) -> None:
        self.project = project
        self.http = http or httpx.Client(transport=httpx.HTTPTransport(uds=SOCKET), base_url="http://docker",
                                         timeout=5.0)  # fmt: skip

    def collect(self) -> Iterator[Metric]:
        """Les métriques des conteneurs du projet."""
        up = GaugeMetricFamily("tennis_docker_up", "1 si l'API Docker a répondu à l'exporteur")
        cpu = CounterMetricFamily("tennis_container_cpu_seconds", "Temps CPU cumulé du conteneur",
                                  labels=["service", "container"])  # fmt: skip
        memory = GaugeMetricFamily("tennis_container_memory_bytes", "Mémoire de travail du conteneur",
                                   labels=["service", "container"])  # fmt: skip
        limit = GaugeMetricFamily("tennis_container_memory_limit_bytes", "Limite mémoire du conteneur",
                                  labels=["service", "container"])  # fmt: skip
        try:
            measures = self.measure()
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            up.add_metric([], 0.0)
            yield up
            return
        up.add_metric([], 1.0)
        for m in measures:
            cpu.add_metric([m.service, m.container], m.cpu_seconds)
            memory.add_metric([m.service, m.container], m.memory_bytes)
            if m.memory_limit_bytes is not None:
                limit.add_metric([m.service, m.container], m.memory_limit_bytes)
        yield from (up, cpu, memory, limit)

    def measure(self) -> list[ContainerStats]:
        """Les conteneurs en marche du projet, et leurs statistiques (lues en parallèle)."""
        filters = json.dumps({"label": [f"{PROJECT_LABEL}={self.project}"], "status": ["running"]})
        response = self.http.get("/containers/json", params={"filters": filters})
        response.raise_for_status()
        listed = [(c["Id"], c.get("Labels", {}).get(SERVICE_LABEL, "?"), str(c["Names"][0]).lstrip("/"))
                  for c in response.json()]  # fmt: skip
        if not listed:
            return []
        with ThreadPoolExecutor(max_workers=min(len(listed), 16)) as pool:
            results = pool.map(lambda item: self._stats(*item), listed)
            return sorted((r for r in results if r is not None), key=lambda r: (r.service, r.container))

    def _stats(self, container_id: str, service: str, name: str) -> ContainerStats | None:
        try:
            response = self.http.get(f"/containers/{container_id}/stats", params={"stream": "false",
                                                                                  "one-shot": "true"})  # fmt: skip
            response.raise_for_status()
            return parse_stats(service, name, response.json())
        except (httpx.HTTPError, ValueError):
            return None  # conteneur arrêté entre la liste et la lecture
