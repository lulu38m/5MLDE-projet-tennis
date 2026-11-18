"""Campagne de charge : profils, analyse, verdicts, rapport (sans lancer Locust ni la stack)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from loadtest.common import RequestLog, match
from loadtest.profiles import DEFAULT_LOAD_CONFIG, LoadConfig, Scenario, Slo, Stage, from_env, ramp, tick, to_env

from loadtest import analysis, run_campaign

SLO = Slo(p95_ms=200, batch_p95_ms=1000, error_rate=0.01)
T0 = 1_000_000.0
Row = tuple[float, str, float, int]


def write_log(directory: Path, rows: list[Row], pid: int = 1) -> None:
    """Un journal de requêtes comme l'écrit locustfile.py."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"requests-{pid}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "name", "ms", "ok"])
        writer.writerows(rows)
    (directory / f"start-{pid}.txt").write_text(f"{T0}\n", encoding="utf-8")


def steady(start: float, seconds: int, rps: int, ms: float, name: str = "/predict", ok: int = 1) -> list[Row]:
    """`rps` requêtes par seconde de durée `ms` pendant `seconds` secondes."""
    return [(start + s + i / rps, name, ms, ok) for s in range(seconds) for i in range(rps)]


# ---------------------------------------------------------------------------- profils


def test_repository_config_is_valid() -> None:
    cfg = LoadConfig.from_yaml(DEFAULT_LOAD_CONFIG)
    assert set(cfg.configs) == {"A", "B", "C"}
    full = cfg.campaigns["full"]
    assert {s.config for s in full if s.name == "ramp"} == {"A", "B", "C"}
    assert {s.kind for s in full} == {"ramp", "spike", "endurance"}
    assert {s.users for s in full} == {"mix", "predict", "batch"}
    ramp_b = next(s for s in full if s.slug == "B-ramp")
    assert [st.users for st in ramp_b.stages()] == [10, 50, 100, 200, 500, 1000]
    assert ramp_b.duration() == 6 * 120
    assert sum(s.duration() for s in cfg.campaigns["quick"]) < 6 * 60  # make load : quelques minutes
    for config in cfg.configs:
        assert (DEFAULT_LOAD_CONFIG.parent.parent / "loadtest" / f"compose.{config}.yml").exists()


def test_ramp_reaches_each_step_within_its_warmup() -> None:
    stages = ramp([10, 50, 100], step_s=120, warmup_s=20)
    assert [s.users for s in stages] == [10, 50, 100]
    for previous, stage in zip([0, 10, 50], stages, strict=True):
        assert (stage.users - previous) / stage.spawn_rate <= 10  # moitié de la montée
        assert stage.measured_from_s == 20


def test_spike_multiplies_users_in_ramp_seconds() -> None:
    scenario = Scenario(name="spike", config="B", base=50, factor=5, ramp_s=10, before_s=60, during_s=90, after_s=120)
    before, peak, after = scenario.stages()
    assert (before.users, peak.users, after.users) == (50, 250, 50)
    assert (peak.users - before.users) / peak.spawn_rate == 10
    assert scenario.kind == "spike"
    assert scenario.duration() == 270


def test_shape_ticks_follow_the_profile_then_stop() -> None:
    stages = [Stage(duration_s=10, users=5, spawn_rate=1), Stage(duration_s=20, users=50, spawn_rate=5)]
    profile = from_env(to_env(stages))
    assert tick(profile, 0) == (5, 1)
    assert tick(profile, 9.9) == (5, 1)
    assert tick(profile, 10) == (50, 5)
    assert tick(profile, 30) is None


def test_scenario_without_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="ni steps"):
        _ = Scenario(name="x", config="A").kind


# ---------------------------------------------------------------------------- locustfile


def test_match_draws_two_distinct_players() -> None:
    pool = [{"id": i, "name": f"P{i}"} for i in range(3)]
    for _ in range(50):
        body = match(pool)
        assert body["player1"]["id"] != body["player2"]["id"]
        assert body["surface"] in {"Hard", "Clay", "Grass"}


def test_request_log_writes_every_row(tmp_path: Path) -> None:
    log = RequestLog(tmp_path)
    for i in range(1203):
        log.add("/predict", float(i), i % 100 != 0)
    log.close()
    frame = analysis.load_requests(tmp_path)
    assert len(frame) == 1203
    assert frame["ok"].sum() == 1203 - 13


# ---------------------------------------------------------------------------- analyse


def test_stage_metrics_are_exact_and_skip_the_warmup(tmp_path: Path) -> None:
    stages = [Stage(duration_s=30, users=10, spawn_rate=10, measured_from_s=10, label="10"),
              Stage(duration_s=30, users=50, spawn_rate=10, measured_from_s=10, label="50")]  # fmt: skip
    rows = steady(T0, 10, 10, 900.0)  # montée lente : ignorée
    rows += steady(T0 + 10, 20, 10, 50.0)
    rows += steady(T0 + 30, 10, 50, 900.0)  # montée du 2e palier : ignorée
    rows += steady(T0 + 40, 20, 50, 150.0) + steady(T0 + 40, 20, 3, 300.0)  # 3/53 lentes > 5 % : p95 = 300
    write_log(tmp_path, rows)
    placed = analysis.windows(stages, analysis.start_time(tmp_path))
    table = analysis.stage_metrics(analysis.load_requests(tmp_path), placed, SLO)
    first, second = table.iloc[0], table.iloc[1]
    assert first["requests"] == 200
    assert first["rps"] == pytest.approx(10, rel=0.02)
    assert first["p95_ms"] == 50
    assert bool(first["slo_ok"])
    assert second["requests"] == 20 * 53
    assert second["p95_ms"] == 300
    assert not bool(second["slo_ok"])


def test_batches_have_their_own_objective(tmp_path: Path) -> None:
    stages = [Stage(duration_s=20, users=10, spawn_rate=10, label="mix")]
    rows = steady(T0, 20, 19, 40.0) + steady(T0, 20, 1, 700.0, name="/predict/batch (50)")
    write_log(tmp_path, rows)
    placed = analysis.windows(stages, T0)
    row = analysis.stage_metrics(analysis.load_requests(tmp_path), placed, SLO).iloc[0]
    assert row["p95_ms"] == 40  # 5 % de batchs ne font pas le p95 des requêtes unitaires
    assert row["batch_p95_ms"] == 700
    assert bool(row["slo_ok"])
    slow = Slo(p95_ms=200, batch_p95_ms=500, error_rate=0.01)
    assert not bool(analysis.stage_metrics(analysis.load_requests(tmp_path), placed, slow).iloc[0]["slo_ok"])


def test_errors_break_the_objective(tmp_path: Path) -> None:
    rows = steady(T0, 10, 98, 20.0) + steady(T0, 10, 2, 20.0, ok=0)
    write_log(tmp_path, rows)
    placed = analysis.windows([Stage(duration_s=10, users=100, spawn_rate=100)], T0)
    row = analysis.stage_metrics(analysis.load_requests(tmp_path), placed, SLO).iloc[0]
    assert row["error_rate"] == pytest.approx(0.02)
    assert not bool(row["slo_ok"])


def _table(users: list[int], ok: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"users": users, "rps": [u * 0.95 for u in users], "p95_ms": [50.0] * len(users),
                         "slo_ok": ok})  # fmt: skip


def test_capacity_is_the_last_step_before_the_first_failure() -> None:
    cap = analysis.capacity(_table([10, 50, 100, 200], [True, True, False, True]))
    assert cap is not None
    assert (cap.users, cap.first_failure) == (50, 100)
    assert cap.rps == pytest.approx(47.5)
    held = analysis.capacity(_table([10, 50], [True, True]))
    assert held is not None
    assert (held.users, held.first_failure) == (50, None)
    assert analysis.capacity(_table([10, 50], [False, True])) is None


def test_spike_recovery_time(tmp_path: Path) -> None:
    stages = [Stage(duration_s=30, users=10, spawn_rate=5, label="avant"),
              Stage(duration_s=30, users=50, spawn_rate=40, label="pic"),
              Stage(duration_s=60, users=10, spawn_rate=40, label="après")]  # fmt: skip
    rows = steady(T0, 30, 10, 30.0) + steady(T0 + 30, 30, 40, 800.0)
    rows += steady(T0 + 60, 20, 10, 600.0) + steady(T0 + 80, 40, 10, 30.0)  # 20 s pour revenir
    write_log(tmp_path, rows)
    requests = analysis.load_requests(tmp_path)
    placed = analysis.windows(stages, T0)
    table = analysis.stage_metrics(requests, placed, SLO)
    result = analysis.spike(table, analysis.timeline(requests, T0), placed, T0, SLO)
    assert result.base_p95_ms == 30
    assert result.peak_p95_ms == 800
    assert result.recovery_s == 20


def test_leak_detection() -> None:
    seconds = np.arange(0, 900, 15, dtype=float)
    flat = pd.DataFrame({"second": seconds, "bytes": 400e6 + np.sin(seconds) * 2e6})
    growing = pd.DataFrame({"second": seconds, "bytes": 400e6 + seconds * 0.1e6})  # +60 Mo / 10 min
    no_leak, leak = analysis.leak(flat), analysis.leak(growing)
    assert no_leak is not None and leak is not None
    assert not no_leak.suspected
    assert leak.suspected
    assert leak.slope_mb_per_10min == pytest.approx(60, rel=0.01)
    assert analysis.leak(flat.head(5)) is None  # tout dans la mise en route


def test_a_frozen_run_is_detected(tmp_path: Path) -> None:
    rows = steady(T0, 59, 10, 20.0) + steady(T0 + 88, 12, 10, 20.0)  # machine en veille entre 59 et 88 s
    write_log(tmp_path, rows)
    pause, at = analysis.longest_pause(analysis.load_requests(tmp_path), T0, T0 + 120)
    assert pause == pytest.approx(29.1, abs=0.2)
    assert at == pytest.approx(58.9, abs=0.2)
    continuous = analysis.longest_pause(analysis.load_requests(tmp_path).query("t < @T0 + 59"), T0, T0 + 59)
    assert continuous[0] < 1


# ---------------------------------------------------------------------------- de bout en bout (sans Locust)


class _NoPrometheus(run_campaign.Prometheus):
    def by_service(self, expr: str, at: float) -> dict[str, float]:
        return {"api": 0.8, "postgres": 0.1} if "cpu" in expr else {"api": 450e6}

    def series(self, expr: str, start: float, end: float, step_s: float = 15) -> pd.DataFrame:
        t = np.arange(start, end, 15.0)
        return pd.DataFrame({"t": t, "value": np.full(len(t), 450e6)})


def test_analyse_and_report_from_saved_logs(tmp_path: Path) -> None:
    cfg = LoadConfig(configs={"A": "petite", "B": "moyenne"}, campaigns={"quick": [
        Scenario(name="ramp", config="A", steps=[5, 10], step_s=20, warmup_s=5),
        Scenario(name="ramp", config="B", steps=[5, 10], step_s=20, warmup_s=5),
        Scenario(name="endurance", config="B", constant=10, duration_s=60, warmup_s=5),
    ]})  # fmt: skip
    prometheus = _NoPrometheus("http://prometheus")
    outcomes = []
    for scenario, slow in zip(cfg.campaigns["quick"], [True, False, False], strict=True):
        raw = tmp_path / scenario.slug / "raw"
        rows = steady(T0, 20, 5, 50.0) + steady(T0 + 20, 20, 10, 400.0 if slow else 60.0)
        if scenario.kind == "endurance":
            rows = steady(T0, 60, 10, 40.0)
        write_log(raw, rows)
        outcomes.append(run_campaign.analyse(scenario, cfg.slo, tmp_path / scenario.slug, prometheus))
    assert [o.capacity.users if o.capacity else None for o in outcomes[:2]] == [5, 10]
    assert outcomes[2].leak is not None and not outcomes[2].leak.suspected
    stages = pd.read_csv(tmp_path / "A-ramp" / "stages.csv")
    assert list(stages["start"]) == [0, 20]  # secondes depuis le début du test
    assert stages["api_cpu_cores"].iloc[0] == pytest.approx(0.8)
    assert (tmp_path / "A-ramp" / "ramp.png").stat().st_size > 10_000
    assert (tmp_path / "B-endurance" / "endurance.png").exists()
    summary = json.loads((tmp_path / "A-ramp" / "summary.json").read_text())
    assert summary["capacity"]["first_failure"] == 10
    text = run_campaign.report(outcomes, cfg, tmp_path, "quick").read_text()
    assert "| ramp (mix) | A : petite | **5** |" in text
    assert "| ramp (mix) | B : moyenne | **10** |" in text
    assert "comparison.png" in text and (tmp_path / "comparison.png").exists()
    assert "pas de fuite" in text
    assert "interrompu" not in text
    # Une rampe qui s'arrête en route est signalée, avec la commande pour la rejouer seule.
    frozen = outcomes[1]
    frozen.pause = (240.0, 100.0)
    text = run_campaign.report(outcomes, cfg, tmp_path, "quick").read_text()
    assert "ramp (mix) ⚠️ interrompu" in text
    assert f"--only B-ramp --out {tmp_path.name}" in text


def test_report_only_rebuilds_the_report_without_replaying(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "load.yaml"
    config.write_text(
        "configs: {B: moyenne}\ncampaigns:\n  quick:\n"
        "    - {name: ramp, config: B, steps: [5, 10], step_s: 20, warmup_s: 5}\n  full: []\n",
        encoding="utf-8",
    )
    out = tmp_path / "campagne"
    write_log(out / "B-ramp" / "raw", steady(T0, 20, 5, 50.0) + steady(T0 + 20, 20, 10, 60.0))
    monkeypatch.setattr(run_campaign, "run_scenario", lambda *a, **k: pytest.fail("rien ne doit être rejoué"))
    argv = [
        "quick",
        "--config",
        str(config),
        "--report-only",
        "--out",
        str(out),
        "--prometheus-url",
        "http://127.0.0.1:9",
    ]
    assert run_campaign.main(argv) == 0
    assert "| ramp (mix) | B : moyenne | **10** |" in (out / "report.md").read_text(encoding="utf-8")
    stages = pd.read_csv(out / "B-ramp" / "stages.csv")
    assert list(stages["ok_rps"]) == pytest.approx([5, 10], rel=0.05)
