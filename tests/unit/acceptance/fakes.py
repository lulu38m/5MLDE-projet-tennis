"""Une stack et des actions en mémoire, pour tester les scénarios et le déroulé sans Docker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from tennis_mlops.acceptance.scenarios import FlowOutcome
from tennis_mlops.acceptance.stack import RunInfo, Stack, TaskInfo, now
from tennis_mlops.config import Settings
from tennis_mlops.orchestration.clock import ClockStep
from tennis_mlops.registry.tagging import Alias


@dataclass
class FakeStack(Stack):
    """Ce que les scénarios lisent, en mémoire ; chaque appel qui change l'état est noté dans `calls`."""

    up: set[str] = field(default_factory=lambda: {"mock_source", "api", "mlflow", "prefect", "alert_sink", "ui"})
    aliases: dict[str, str] = field(default_factory=lambda: {"production": "1"})
    tags: dict[str, dict[str, str]] = field(default_factory=lambda: {"1": {"lifecycle": "production"}})
    served: str | None = "1"
    reloads: int = 3
    now_day: date = date(2016, 1, 1)
    mark: date = date(2016, 1, 1)
    chaos_mode: str = "off"
    alert_list: list[dict[str, Any]] = field(default_factory=list)
    event_list: list[dict[str, Any]] = field(default_factory=list)
    runs: list[RunInfo] = field(default_factory=list)
    tasks: dict[str, list[TaskInfo]] = field(default_factory=dict)
    silenced: bool = False
    teams: bool = True
    paused: bool = True
    calls: list[str] = field(default_factory=list)
    sql: dict[str, Any] = field(default_factory=dict)  # début de requête → résultat

    def __post_init__(self) -> None:
        self.settings = Settings(_env_file=None)

    def close(self) -> None:
        pass

    # santé
    def healthy(self, service: str) -> bool:
        return service in self.up

    # mock_source
    def clock(self) -> date:
        return self.now_day

    def set_clock(self, day: date) -> None:
        self.calls.append(f"clock={day}")
        self.now_day = day

    def advance(self, days: int) -> ClockStep:
        previous, self.now_day = self.now_day, self.now_day + timedelta(days=days)
        return ClockStep(previous, self.now_day, 40, ["Doha"])

    def chaos(self, mode: str, failures: int | None = None, delay_s: float | None = None) -> dict[str, Any]:
        self.chaos_mode = mode
        self.calls.append(f"chaos={mode}")
        return {"mode": mode}

    # API
    def ready(self) -> tuple[int, dict[str, Any]]:
        status = "degraded" if "mlflow" not in self.up else "ready"
        reasons = ["registre MLflow injoignable, v1 servie depuis le cache"] if status == "degraded" else []
        return 200, {"status": status, "checks": {"features_last_day": "2015-01-31"}, "reasons": reasons}

    def model_info(self) -> dict[str, Any]:
        return {"served": {"version": self.served} if self.served else None, "registry": {"reloads": self.reloads}}

    def predict(self, body: Mapping[str, Any]) -> httpx.Response:
        if body.get("surface") == "Moon":
            return httpx.Response(422, headers={"x-request-id": "bad-1"}, json={"error": "validation_error"})
        return httpx.Response(200, headers={"x-model-version": str(self.served)},
                              json={"player1": {"win_probability": 0.71}})  # fmt: skip

    # alertes
    def alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self.alert_list)

    def alert_health(self) -> dict[str, Any]:
        return {"teams": self.teams, "silence": {"active": self.silenced}}

    def silence(self, minutes: int, reason: str) -> dict[str, Any]:
        self.silenced = True
        self.calls.append("silence")
        return {"active": True}

    def unsilence(self) -> None:
        self.silenced = False
        self.calls.append("unsilence")

    # registre
    def alias(self, alias: Alias) -> str | None:
        return self.aliases.get(alias.value)

    def version_tags(self, version: str) -> dict[str, str]:
        return dict(self.tags.get(version, {}))

    def latest_version(self) -> str | None:
        return max(self.tags, key=int) if self.tags else None

    def version_count(self) -> int:
        return len(self.tags)

    # base
    def rows(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        return list(self._sql(sql, []))

    def scalar(self, sql: str, **params: Any) -> Any:
        return self._sql(sql, 0)

    def _sql(self, sql: str, default: Any) -> Any:
        for prefix, value in self.sql.items():
            if sql.startswith(prefix):
                return value() if callable(value) else value
        return default

    def watermark(self) -> date:
        return self.mark

    def events(self, kind: str, since: datetime) -> list[dict[str, Any]]:
        return [e for e in self.event_list if e["kind"] == kind and e["occurred_at"] >= since]

    # Prefect
    def flow_runs(self, flow_names: Any = (), since: datetime | None = None) -> list[RunInfo]:
        names = set(flow_names)
        return [
            r
            for r in self.runs
            if (not names or r.flow_name in names) and (since is None or (r.created or now()) >= since)
        ]

    def task_runs(self, flow_run_id: str) -> list[TaskInfo]:
        return list(self.tasks.get(flow_run_id, []))

    def pause_simulation(self) -> bool:
        was_running, self.paused = not self.paused, True
        return was_running

    # helpers de test
    def event(self, kind: str, **fields: Any) -> None:
        self.event_list.append({"kind": kind, "occurred_at": now(), "title": kind, "flow_run_id": None,
                                "lot_id": None, "payload": {}, **fields})  # fmt: skip

    def training(self, name: str = "sage-heron", state: str = "COMPLETED") -> RunInfo:
        run = RunInfo(f"run-{len(self.runs)}", name, "entrainement-atp", state, state.title(), now())
        self.runs.append(run)
        return run


class FakeServices:
    """`docker compose stop|start` noté, et l'effet sur la fausse stack."""

    def __init__(self, stack: FakeStack) -> None:
        self.stack = stack
        self.calls: list[str] = []

    def stop(self, service: str) -> None:
        self.calls.append(f"stop {service}")
        self.stack.up.discard(service)

    def start(self, service: str) -> None:
        self.calls.append(f"start {service}")
        self.stack.up.add(service)
        if service == "mock_source":
            self.stack.now_day = date(2016, 1, 1)  # l'horloge vit en mémoire : un redémarrage la remet au départ


@dataclass
class FakeActions:
    """Les commandes du projet, réduites à leur effet sur la fausse stack."""

    stack: FakeStack
    ingests: list[FlowOutcome] = field(default_factory=list)
    simulations: list[FlowOutcome] = field(default_factory=list)
    on_ingest: Any = None  # effet de bord d'une ingestion (ex. événements)
    on_simulate: Any = None  # effet de bord d'un pas de simulation (entraînement déclenché…)
    train_code: int = 0
    rollback_code: int = 0
    smoke: list[str] = field(default_factory=lambda: ["interface en ligne", "prédiction affichée", "tracée"])
    calls: list[str] = field(default_factory=list)

    def ingest(self) -> FlowOutcome:
        self.calls.append(f"ingest chaos={self.stack.chaos_mode}")
        if self.on_ingest:
            self.on_ingest(self)
        return self.ingests.pop(0) if self.ingests else FlowOutcome("ing-ok", "Completed", False)

    def simulate(self, days: int) -> FlowOutcome:
        self.calls.append(f"simulate {days}")
        if self.on_simulate:
            self.on_simulate(self, days)
        return self.simulations.pop(0)

    def train_weak(self) -> int:
        self.calls.append("train-weak")
        version = str(int(self.stack.latest_version() or "0") + 1)
        self.stack.tags[version] = {"gate_status": "failed", "lifecycle": "rejected", "gate_reason": "log loss"}
        self.stack.aliases["staging"] = version
        return self.train_code

    def rollback(self, reason: str) -> int:
        self.calls.append("rollback")
        production, previous = self.stack.aliases["production"], self.stack.aliases["previous"]
        self.stack.aliases["production"] = previous
        self.stack.tags[production]["lifecycle"] = "rolled_back"
        self.stack.served, self.stack.reloads = previous, self.stack.reloads + 1
        self.stack.alert_list.insert(0, {"kind": "rollback", "title": "↩️ Rollback", "delivery": "silenced",
                                         "at": now().isoformat(), "flow_run_id": None})  # fmt: skip
        return self.rollback_code

    def ui_smoke(self, ui_url: str, settings: Settings) -> list[str]:
        self.calls.append("ui-smoke")
        return self.smoke
