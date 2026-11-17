"""Exporteur Prometheus de la chaîne : ce que l'API ne mesure pas (les conteneurs : `containers.py`).

À chaque collecte (toutes les 15 s, `infra/prometheus/prometheus.yml`), il interroge :

- **la santé** de MLflow, de la source (`mock_source`), de Prefect, de l'API et du service d'alertes
  (`tennis_service_up`, `tennis_service_probe_seconds`) ;
- **la source** : horloge simulée et mode d'incident en cours (`tennis_simulated_clock_timestamp`,
  `tennis_source_chaos`), pour lire les périodes de chaos sur les dashboards ;
- **Prefect** : runs des dernières 24 h par flow et par état, durée du dernier run terminé, relances de tasks et de
  flows (`tennis_prefect_*`).

Un service injoignable n'interrompt pas la collecte : il est compté comme arrêté. Les sondes partent en parallèle
(3 s au plus chacune) ; les réponses de Prefect sont gardées 10 s, pour ne pas le solliciter plus que Prometheus ne
lit.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

CHAOS_MODES = ("off", "503", "slow", "corrupt", "schema")
WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class Targets:
    """Les adresses sondées (réseau interne de la stack, ou machine hôte)."""

    mlflow: str
    mock_source: str
    prefect_api: str
    api: str
    alert_sink: str

    def probes(self) -> dict[str, str]:
        """Le chemin de santé de chaque service."""
        return {
            "mlflow": f"{self.mlflow}/health", "mock_source": f"{self.mock_source}/health",
            "prefect": f"{self.prefect_api}/health", "api": f"{self.api}/health",
            "alert_sink": f"{self.alert_sink}/health",
        }  # fmt: skip


@dataclass
class PrefectSummary:
    """Les runs des dernières 24 h, réduits à ce que les dashboards montrent."""

    runs: Counter[tuple[str, str]] = field(default_factory=Counter)  # (flow, état) → nombre
    last_seconds: dict[str, float] = field(default_factory=dict)  # flow → durée du dernier run terminé
    last_end: dict[str, float] = field(default_factory=dict)  # flow → fin du dernier run terminé (timestamp)
    retries: Counter[str] = field(default_factory=Counter)  # flow → relances (tasks et flow)


class ChainCollector(Collector):
    """Collecteur Prometheus : une passe de sondes à chaque lecture."""

    def __init__(self, targets: Targets, http: httpx.Client | None = None, clock: Callable[[], float] = time.time,
                 cache_s: float = 10.0) -> None:  # fmt: skip
        self.targets = targets
        self.http = http or httpx.Client(timeout=3.0)
        self.clock = clock
        self.cache_s = cache_s
        self._prefect: tuple[float, PrefectSummary | None] | None = None

    def collect(self) -> Iterator[Metric]:
        """Les métriques, recalculées à chaque lecture."""
        up = GaugeMetricFamily("tennis_service_up", "1 si le service répond à sa sonde de santé", labels=["service"])
        seconds = GaugeMetricFamily("tennis_service_probe_seconds", "Durée de la sonde", labels=["service"])
        source_health: dict[str, Any] | None = None
        probes = self.targets.probes()
        with ThreadPoolExecutor(max_workers=len(probes)) as pool:
            results = dict(zip(probes, pool.map(self._probe, probes.values()), strict=True))
        for service, (ok, elapsed, body) in results.items():
            up.add_metric([service], 1.0 if ok else 0.0)
            seconds.add_metric([service], elapsed)
            if service == "mock_source" and ok:
                source_health = body
        yield up
        yield seconds
        yield from self._source(source_health)
        yield from self._prefect_metrics()

    # ------------------------------------------------------------------ santé et source

    def _probe(self, url: str) -> tuple[bool, float, dict[str, Any] | None]:
        started = time.perf_counter()
        try:
            response = self.http.get(url)
        except httpx.HTTPError:
            return False, time.perf_counter() - started, None
        elapsed = time.perf_counter() - started
        try:
            body = response.json()
        except ValueError:
            body = None
        return response.status_code == 200, elapsed, body if isinstance(body, dict) else None

    def _source(self, health: dict[str, Any] | None) -> Iterator[Metric]:
        clock = GaugeMetricFamily("tennis_simulated_clock_timestamp", "Date simulée de la source (timestamp)")
        chaos = GaugeMetricFamily("tennis_source_chaos", "1 pour le mode d'incident en cours de la source",
                                  labels=["mode"])  # fmt: skip
        if health is not None:
            now = health.get("simulated_now")
            if now:
                clock.add_metric([], _timestamp(str(now)))
            current = str(health.get("chaos_mode") or "off")
            for mode in dict.fromkeys((*CHAOS_MODES, current)):
                chaos.add_metric([mode], 1.0 if mode == current else 0.0)
        yield clock
        yield chaos

    # ------------------------------------------------------------------ Prefect

    def _prefect_metrics(self) -> Iterator[Metric]:
        summary = self._summary()
        reachable = GaugeMetricFamily("tennis_exporter_prefect_up", "1 si l'API de Prefect a répondu à l'exporteur")
        reachable.add_metric([], 0.0 if summary is None else 1.0)
        yield reachable
        runs = GaugeMetricFamily("tennis_prefect_flow_runs", "Runs commencés dans les dernières 24 h",
                                 labels=["flow", "state"])  # fmt: skip
        last = GaugeMetricFamily("tennis_prefect_flow_run_seconds", "Durée du dernier run terminé", labels=["flow"])
        ended = GaugeMetricFamily("tennis_prefect_flow_last_end_timestamp", "Fin du dernier run terminé",
                                  labels=["flow"])  # fmt: skip
        retries = GaugeMetricFamily("tennis_prefect_retries", "Relances (tasks et flows) dans les dernières 24 h",
                                    labels=["flow"])  # fmt: skip
        if summary is not None:
            for (flow, state), count in sorted(summary.runs.items()):
                runs.add_metric([flow, state], float(count))
            for flow, value in sorted(summary.last_seconds.items()):
                last.add_metric([flow], value)
            for flow, value in sorted(summary.last_end.items()):
                ended.add_metric([flow], value)
            for flow in sorted({f for f, _ in summary.runs}):
                retries.add_metric([flow], float(summary.retries.get(flow, 0)))
        yield runs
        yield last
        yield ended
        yield retries

    def _summary(self) -> PrefectSummary | None:
        now = self.clock()
        if self._prefect is not None and now - self._prefect[0] < self.cache_s:
            return self._prefect[1]
        try:
            summary: PrefectSummary | None = self._read_prefect(now)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            summary = None
        self._prefect = (now, summary)
        return summary

    def _read_prefect(self, now: float) -> PrefectSummary:
        api = self.targets.prefect_api
        since = (datetime.fromtimestamp(now, UTC) - WINDOW).isoformat()
        flows = {f["id"]: f["name"] for f in self._post(f"{api}/flows/filter", {"limit": 200})}
        recent = {"start_time": {"after_": since}}
        flow_runs = self._post(f"{api}/flow_runs/filter",
                               {"flow_runs": recent, "sort": "START_TIME_DESC", "limit": 200})  # fmt: skip
        task_runs = self._post(f"{api}/task_runs/filter",
                               {"task_runs": recent, "sort": "EXPECTED_START_TIME_DESC", "limit": 200})  # fmt: skip
        summary = PrefectSummary()
        run_flow: dict[str, str] = {}
        for run in flow_runs:  # du plus récent au plus ancien
            flow = flows.get(run.get("flow_id"), "inconnu")
            run_flow[run["id"]] = flow
            state = run.get("state_name") or run.get("state_type") or "inconnu"
            summary.runs[(flow, str(state))] += 1
            summary.retries[flow] += max(int(run.get("run_count") or 0) - 1, 0)
            finished = run.get("end_time") and run.get("total_run_time") is not None
            if flow not in summary.last_seconds and finished:
                summary.last_seconds[flow] = float(run["total_run_time"])
                summary.last_end[flow] = _timestamp(str(run["end_time"]))
        for task in task_runs:
            flow = run_flow.get(str(task.get("flow_run_id")), "inconnu")
            summary.retries[flow] += max(int(task.get("run_count") or 0) - 1, 0)
        return summary

    def _post(self, url: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        response = self.http.post(url, json=body)
        response.raise_for_status()
        data = response.json()
        return list(data) if isinstance(data, list) else []


def _timestamp(value: str) -> float:
    """Un horodatage ISO (ou une date seule) en secondes UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()
