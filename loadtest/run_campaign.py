"""Campagne de charge automatisée : `python -m loadtest.run_campaign quick|full` (`make load`, `make load-full`).

Pour chaque scénario de `configs/load.yaml` :

1. l'API est remise dans la configuration voulue (A, B ou C : `docker compose -f compose.yml -f
   loadtest/compose.<X>.yml up`), puis on attend qu'elle soit prête ;
2. Locust tourne sans interface (`--headless`, plusieurs processus), en suivant les paliers du scénario ; chaque
   requête est journalisée ;
3. on calcule par palier débit, erreurs et percentiles exacts, et on lit dans Prometheus ce que l'API a consommé
   (CPU et mémoire de ses conteneurs, via l'exporteur de la phase 9) ; le CPU de Locust est mesuré à part (il tourne
   sur la même machine) ;
4. CSV, graphiques et un rapport Markdown sont écrits dans `loadtest/results/<date>-<campagne>/`.

À la fin, l'API est remise dans sa configuration normale (celle de `compose.yml`).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import psutil

from loadtest import analysis, charts
from loadtest.profiles import DEFAULT_LOAD_CONFIG, USER_CLASSES, LoadConfig, Scenario, Slo, to_env

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "loadtest" / "results"
LOCUSTFILES = "loadtest/locustfile.py,loadtest/shape.py"
MAX_PAUSE_S = 10  # au-delà, aucune requête pendant ce temps : le test a été interrompu


# ---------------------------------------------------------------------------- la stack


@dataclass
class Stack:
    """La configuration de l'API sous test (compose), et son adresse."""

    api_url: str
    prometheus_url: str
    manage: bool = True  # False : on teste l'API telle qu'elle tourne (pas de docker compose)
    current: str | None = None

    def use(self, config: str) -> None:
        """Remet l'API dans la configuration `config`, puis attend qu'elle soit prête."""
        if not self.manage or self.current == config:
            return
        print(f"→ configuration {config} de l'API…", flush=True)
        if config != "C":  # l'équilibreur publie le port de l'API : il doit partir avant que l'API le reprenne
            self._compose(["loadtest/compose.C.yml"], ["rm", "-sf", "api-lb"], check=False)
        services = ["api", "api-lb"] if config == "C" else ["api"]
        self._compose([f"loadtest/compose.{config}.yml"], ["up", "-d", "--wait", "--remove-orphans", *services])
        self.current = config
        self.wait_ready()

    def restore(self) -> None:
        """L'API telle que `compose.yml` la décrit."""
        if not self.manage or self.current is None:
            return
        print("→ retour de l'API à sa configuration normale…", flush=True)
        self._compose(["loadtest/compose.C.yml"], ["rm", "-sf", "api-lb"], check=False)
        self._compose([], ["up", "-d", "--wait", "--remove-orphans", "api"])
        self.current = None

    def wait_ready(self, timeout_s: float = 240) -> None:
        """`GET /ready` répond 200 (modèle et historique chargés)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.api_url}/ready", timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3)
        raise RuntimeError(f"l'API ({self.api_url}) n'est pas prête après {timeout_s:.0f} s")

    def _compose(self, overrides: list[str], args: list[str], check: bool = True) -> None:
        """`docker compose -f compose.yml [-f <surcharge>…] <args>`, depuis la racine du dépôt."""
        files = [x for f in ["compose.yml", *overrides] for x in ("-f", f)]
        result = subprocess.run(["docker", "compose", *files, *args], cwd=ROOT, capture_output=True, text=True,
                                check=False)  # fmt: skip
        if check and result.returncode != 0:
            raise RuntimeError(f"docker compose {' '.join(args)} : {result.stderr.strip()[-800:]}")


# ---------------------------------------------------------------------------- Prometheus


@dataclass
class Prometheus:
    """Lectures dans Prometheus (CPU et mémoire des conteneurs, exporteur de la phase 9)."""

    url: str

    def by_service(self, expr: str, at: float) -> dict[str, float]:
        """Une requête instantanée groupée par `service`."""
        try:
            response = httpx.get(f"{self.url}/api/v1/query", params={"query": expr, "time": at}, timeout=10)
            response.raise_for_status()
            result = response.json()["data"]["result"]
        except (httpx.HTTPError, ValueError, KeyError):
            return {}
        return {r["metric"].get("service", ""): float(r["value"][1]) for r in result}

    def series(self, expr: str, start: float, end: float, step_s: float = 15) -> pd.DataFrame:
        """Une série dans le temps (colonnes `t`, `value`)."""
        try:
            params: dict[str, str | float] = {"query": expr, "start": start, "end": end, "step": step_s}
            response = httpx.get(f"{self.url}/api/v1/query_range", params=params, timeout=10)
            response.raise_for_status()
            result = response.json()["data"]["result"]
        except (httpx.HTTPError, ValueError, KeyError):
            return pd.DataFrame(columns=["t", "value"])
        if not result:
            return pd.DataFrame(columns=["t", "value"])
        return pd.DataFrame([(float(t), float(v)) for t, v in result[0]["values"]], columns=["t", "value"])

    def resources(self, start: float, end: float) -> dict[str, float]:
        """CPU moyen (cœurs) et mémoire maximale (Mo) par service, entre `start` et `end`."""
        seconds = max(int(end - start), 15)
        used = f"sum by (service) (increase(tennis_container_cpu_seconds_total[{seconds}s])) / {seconds}"
        cpu = self.by_service(used, end)
        memory = self.by_service(
            f"max_over_time(sum by (service) (tennis_container_memory_bytes)[{seconds}s:15s])", end
        )
        out = {f"{s}_cpu_cores": v for s, v in cpu.items() if s in ("api", "api-lb", "postgres")}
        out |= {f"{s}_memory_mb": v / 1e6 for s, v in memory.items() if s in ("api", "api-lb", "postgres")}
        return out


# ---------------------------------------------------------------------------- CPU de Locust


@dataclass
class CpuSampler:
    """CPU consommé par Locust (processus et enfants), échantillonné toutes les 2 s."""

    pid: int
    samples: list[tuple[float, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    # Un même objet par processus : cpu_percent() mesure depuis l'appel précédent sur cet objet (0 au premier).
    _known: dict[int, psutil.Process] = field(default_factory=dict)

    def start(self) -> None:
        """Démarre l'échantillonnage (thread)."""
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        """Arrête l'échantillonnage."""
        self._stop.set()

    def _run(self) -> None:
        try:
            root = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop.is_set():
            try:
                current = [root, *root.children(recursive=True)]
            except psutil.Error:
                break
            total = 0.0
            for p in current:
                known = self._known.setdefault(p.pid, p)
                try:
                    total += known.cpu_percent(interval=None)
                except psutil.Error:
                    continue
            self.samples.append((time.time(), total / 100))  # en cœurs
            self._stop.wait(2)


# ---------------------------------------------------------------------------- un scénario


@dataclass
class Outcome:
    """Ce qu'un scénario a donné."""

    scenario: Scenario
    directory: Path
    stages: pd.DataFrame
    capacity: analysis.Capacity | None = None
    spike: analysis.SpikeResult | None = None
    leak: analysis.LeakResult | None = None
    pause: tuple[float, float] = (0.0, 0.0)  # (durée, début) du plus long moment sans requête

    @property
    def interrupted(self) -> bool:
        """Locust s'est arrêté en route : mesures incomplètes, scénario à relancer."""
        return self.pause[0] > MAX_PAUSE_S


def run_scenario(scenario: Scenario, cfg: LoadConfig, stack: Stack, prometheus: Prometheus, out: Path) -> Outcome:
    """Lance Locust sur un scénario et analyse ce qu'il a mesuré."""
    stack.use(scenario.config)
    directory = out / scenario.slug
    raw = directory / "raw"
    if directory.exists():
        shutil.rmtree(directory)
    raw.mkdir(parents=True)
    stages = scenario.stages()
    minutes = scenario.duration() / 60
    print(f"→ {scenario.slug} : {len(stages)} palier(s), {minutes:.1f} min ({scenario.users})", flush=True)
    env = os.environ | {
        "LOAD_PROFILE": to_env(stages), "LOAD_LOG_DIR": str(raw), "LOAD_TIMEOUT_S": str(cfg.locust.timeout_s),
        "LOAD_PLAYERS": str(cfg.locust.players), "PYTHONPATH": str(ROOT),
    }  # fmt: skip
    command = [sys.executable, "-m", "locust", "-f", LOCUSTFILES, "--headless", "--host", stack.api_url,
               "--processes", str(cfg.locust.processes), "--csv", str(raw / "locust"), "--only-summary",
               "--loglevel", "WARNING", USER_CLASSES[scenario.users]]  # fmt: skip
    with (raw / "locust.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        sampler = CpuSampler(process.pid)
        sampler.start()
        process.wait(timeout=scenario.duration() + 300)
        sampler.stop()
    pd.DataFrame(sampler.samples, columns=["t", "cores"]).to_csv(raw / "locust_cpu.csv", index=False)
    time.sleep(16)  # une collecte Prometheus de plus après la fin
    return analyse(scenario, cfg.slo, directory, prometheus)


def _locust_cpu(directory: Path, placed: list[analysis.Window]) -> list[float]:
    """CPU de Locust par palier : échantillons de la campagne, sinon la colonne d'une analyse précédente."""
    samples = directory / "raw" / "locust_cpu.csv"
    if samples.exists():
        frame = pd.read_csv(samples)
        return [float(frame[(frame["t"] >= w.measure_from) & (frame["t"] <= w.end)]["cores"].mean()) for w in placed]
    previous = directory / "stages.csv"
    if previous.exists():
        old = pd.read_csv(previous)
        if "locust_cpu_cores" in old and len(old) == len(placed):
            return [float(v) for v in old["locust_cpu_cores"]]
    return [float("nan")] * len(placed)


def analyse(scenario: Scenario, slo: Slo, directory: Path, prometheus: Prometheus) -> Outcome:
    """Mesures par palier, chronologie, verdicts, fichiers et graphiques d'un scénario déjà joué."""
    raw = directory / "raw"
    requests = analysis.load_requests(raw)
    t0 = analysis.start_time(raw)
    placed = analysis.windows(scenario.stages(), t0)
    stages = analysis.stage_metrics(requests, placed, slo)
    extra = []
    for w, locust in zip(placed, _locust_cpu(directory, placed), strict=True):
        row = prometheus.resources(w.measure_from, w.end)
        row["locust_cpu_cores"] = locust
        extra.append(row)
    stages = pd.concat([stages, pd.DataFrame(extra)], axis=1)
    line = analysis.timeline(requests, t0)
    saved = stages.copy()
    for column in ("start", "measure_from", "end"):  # en secondes depuis le début du test (t0 dans summary.json)
        saved[column] = saved[column] - t0
    saved.to_csv(directory / "stages.csv", index=False, float_format="%.4g")
    analysis.endpoint_metrics(requests, placed).to_csv(directory / "endpoints.csv", index=False, float_format="%.4g")
    line.to_csv(directory / "timeline.csv", index=False, float_format="%.4g")
    users = pd.DataFrame({"second": [w.start - t0 for w in placed], "users": [w.stage.users for w in placed]})
    outcome = Outcome(scenario, directory, stages, pause=analysis.longest_pause(requests, t0, placed[-1].end))
    title = f"{scenario.slug} · {scenario.users}"
    if scenario.kind == "ramp":
        outcome.capacity = analysis.capacity(stages)
        charts.ramp(stages, title, slo, directory / "ramp.png")
    elif scenario.kind == "spike":
        outcome.spike = analysis.spike(stages, line, placed, t0, slo)
        charts.timeline(line, users, title, slo, directory / "spike.png")
    else:
        memory = prometheus.series('sum(tennis_container_memory_bytes{service="api"})', placed[0].start, placed[-1].end)
        memory = pd.DataFrame({"second": memory["t"] - t0, "bytes": memory["value"]})
        memory.to_csv(directory / "memory.csv", index=False, float_format="%.6g")
        outcome.leak = analysis.leak(memory, skip_s=scenario.warmup_s)
        charts.timeline(line, users, title, slo, directory / "endurance.png", memory)
    summary = {"scenario": scenario.model_dump(), "kind": scenario.kind, "t0": t0, "requests": len(requests),
               "capacity": asdict(outcome.capacity) if outcome.capacity else None,
               "spike": asdict(outcome.spike) if outcome.spike else None,
               "leak": asdict(outcome.leak) if outcome.leak else None,
               "longest_pause_s": outcome.pause[0], "interrupted": outcome.interrupted}  # fmt: skip
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return outcome


# ---------------------------------------------------------------------------- rapport


def machine() -> dict[str, str]:
    """La machine de référence (le Mac et ce qui est alloué à Docker)."""
    info = {
        "système": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cœurs (machine)": str(psutil.cpu_count()),
        "mémoire (machine)": f"{psutil.virtual_memory().total / 2**30:.0f} Go",
    }
    if platform.system() == "Darwin":
        model = subprocess.run(["sysctl", "-n", "hw.model", "machdep.cpu.brand_string"], capture_output=True,
                               text=True, check=False).stdout.split("\n")  # fmt: skip
        info["modèle"] = " · ".join(x for x in model if x)
    docker = subprocess.run(["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}} {{.ServerVersion}}"],
                            capture_output=True, text=True, check=False).stdout.split()  # fmt: skip
    if len(docker) == 3:
        info["Docker"] = f"{docker[2]} · {docker[0]} CPU · {int(docker[1]) / 2**30:.1f} Go alloués"
    return info


def _fmt(value: Any, unit: str = "", digits: int = 0) -> str:  # noqa: ANN401
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "—"
    return f"{number:,.{digits}f}{unit}".replace(",", " ")


def report(outcomes: list[Outcome], cfg: LoadConfig, out: Path, name: str) -> Path:
    """`report.md` : machine, objectif, capacité par configuration, pic, endurance, biais."""
    slo = cfg.slo
    lines = [f"# Campagne de charge « {name} » ({datetime.now():%d/%m/%Y %H:%M})", "",
             f"Objectif d'un palier : **p95 < {slo.p95_ms:.0f} ms** sur les requêtes unitaires, **p95 < "
             f"{slo.batch_p95_ms:.0f} ms** sur les batchs de 50 matchs, et **erreurs < {slo.error_rate:.0%}**. Un "
             "utilisateur envoie environ une requête par seconde.", "", "## Machine de référence", ""]  # fmt: skip
    lines += [f"- {k} : {v}" for k, v in machine().items()]
    lines += ["", "## Capacité (rampes)", "",
              "| Scénario | Configuration | Max utilisateurs tenus | Débit (req/s) | p95 (ms) | Premier palier qui "
              "lâche | CPU API (cœurs) | Mémoire API (Mo) | CPU Locust (cœurs) |",
              "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    ramps: dict[str, pd.DataFrame] = {}
    for o in outcomes:
        if o.scenario.kind != "ramp":
            continue
        if o.scenario.users == "mix":
            ramps[o.scenario.config] = o.stages
        cap = o.capacity
        at = o.stages[o.stages["users"] == cap.users].iloc[0] if cap else None
        lines.append(" | ".join([
            f"| {o.scenario.name} ({o.scenario.users}){' ⚠️ interrompu' if o.interrupted else ''}",
            f"{o.scenario.config} : {cfg.configs.get(o.scenario.config, '')}",
            f"**{cap.users}**" if cap else "aucun", _fmt(cap.rps if cap else None), _fmt(cap.p95_ms if cap else None),
            str(cap.first_failure) if cap and cap.first_failure else ("—" if cap else _fmt(o.stages['users'].iloc[0])),
            _fmt(at.get("api_cpu_cores") if at is not None else None, digits=2),
            _fmt(at.get("api_memory_mb") if at is not None else None),
            _fmt(at.get("locust_cpu_cores") if at is not None else None, digits=2),
        ]) + " |")  # fmt: skip
    if len(ramps) > 1:
        charts.compare(ramps, slo, out / "comparison.png")
        lines += ["", "![Comparaison des configurations](comparison.png)"]
    for o in outcomes:
        rel = o.directory.relative_to(out)
        if o.spike:
            s = o.spike
            back = f"{s.recovery_s:.0f} s" if s.recovery_s is not None else "pas revenu sous l'objectif"
            lines += ["", f"## Pic ({o.scenario.slug})", "",
                      f"- p95 avant : {_fmt(s.base_p95_ms)} ms · pendant : **{_fmt(s.peak_p95_ms)} ms** "
                      f"(erreurs {s.peak_error_rate:.1%}) · après : {_fmt(s.after_p95_ms)} ms",
                      f"- retour sous l'objectif après la fin du pic : **{back}**", "",
                      f"![Pic]({rel}/spike.png)"]  # fmt: skip
        if o.scenario.kind == "endurance":
            k = o.leak
            if k is None:
                memory = "- mémoire de l'API : non mesurée (Prometheus ou l'exporteur ne répondait pas)"
            else:
                verdict = "**fuite suspectée**" if k.suspected else "pas de fuite"
                memory = (f"- mémoire de l'API : {_fmt(k.start_mb)} → {_fmt(k.end_mb)} Mo, pente "
                          f"{_fmt(k.slope_mb_per_10min, digits=1)} Mo / 10 min : {verdict}")  # fmt: skip
            lines += ["", f"## Endurance ({o.scenario.slug})", "", memory,
                      f"- {o.scenario.constant} utilisateurs pendant {o.scenario.duration_s / 60:.0f} min : p95 "
                      f"{_fmt(o.stages['p95_ms'].iloc[0])} ms, erreurs {o.stages['error_rate'].iloc[0]:.2%}", "",
                      f"![Endurance]({rel}/endurance.png)"]  # fmt: skip
    broken = [o for o in outcomes if o.interrupted]
    if broken:
        lines += ["", "## ⚠️ Scénarios interrompus", ""]
        lines += [f"- `{o.scenario.slug}` : aucune requête pendant {o.pause[0]:.0f} s à partir de {o.pause[1]:.0f} s "
                  f"(machine en veille ? Locust figé ?) : mesures incomplètes, à relancer avec "
                  f"`--only {o.scenario.slug} --out {out.name}`" for o in broken]  # fmt: skip
    lines += ["", "## Détail par scénario", ""]
    lines += [f"- `{o.directory.relative_to(out)}/` : stages.csv, endpoints.csv, timeline.csv" for o in outcomes]
    lines += ["", "## Biais", "",
              "Locust tourne sur la même machine que la stack : il prend du CPU à l'API (colonne CPU Locust). Les "
              "capacités mesurées sont donc un minimum ; sur deux machines, l'API en tiendrait davantage."]  # fmt: skip
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------- point d'entrée


def keep_awake() -> None:
    """Sur macOS, empêche la mise en veille tant que la campagne tourne (`caffeinate`, rendu à la fin).

    Une veille fige Locust et la VM de Docker : la première campagne a perdu ainsi sa rampe B.
    """
    if platform.system() == "Darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])


def main(argv: list[str] | None = None) -> int:
    """`quick` ou `full` (ou `--only B-ramp,B-spike`)."""
    parser = argparse.ArgumentParser(prog="python -m loadtest.run_campaign", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("campaign", choices=["quick", "full"])
    parser.add_argument("--only", help="scénarios à jouer, par nom de dossier (ex. B-ramp,B-spike)")
    parser.add_argument("--config", type=Path, default=DEFAULT_LOAD_CONFIG)
    parser.add_argument("--api-url", default=os.environ.get("TENNIS_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--prometheus-url", default=os.environ.get("TENNIS_PROMETHEUS_URL", "http://127.0.0.1:9091"))
    parser.add_argument("--no-compose", action="store_true", help="tester l'API telle qu'elle tourne")
    parser.add_argument("--out", type=Path, help="dossier de résultats (par défaut loadtest/results/<date>-<nom>) ; "
                        "un dossier existant garde ses autres scénarios, réanalysés pour le rapport")  # fmt: skip
    replay = "ne rien rejouer : réanalyser les scénarios du dossier --out et refaire le rapport"
    parser.add_argument("--report-only", action="store_true", help=replay)
    args = parser.parse_args(argv)
    if args.report_only and not args.out:
        parser.error("--report-only demande --out <dossier de la campagne>")
    keep_awake()
    cfg = LoadConfig.from_yaml(args.config)
    scenarios = cfg.campaigns[args.campaign]
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s.slug in wanted]
    if args.report_only:
        scenarios = []
    out = args.out or RESULTS / f"{datetime.now():%Y%m%d-%H%M}-{args.campaign}"
    out.mkdir(parents=True, exist_ok=True)
    total = sum(s.duration() for s in scenarios) / 60
    print(f"Campagne {args.campaign} : {len(scenarios)} scénario(s), ~{total:.0f} min de charge → {out}", flush=True)
    stack = Stack(args.api_url.rstrip("/"), args.prometheus_url.rstrip("/"), manage=not args.no_compose)
    prometheus = Prometheus(stack.prometheus_url)
    played: dict[str, Outcome] = {}
    try:
        if not stack.manage:
            stack.wait_ready()
        for scenario in scenarios:
            outcome = run_scenario(scenario, cfg, stack, prometheus, out)
            cap = outcome.capacity
            note = f" : {cap.users} utilisateurs tenus" if cap else ""
            warn = f" ⚠️ interrompu ({outcome.pause[0]:.0f} s sans requête) : à relancer" if outcome.interrupted else ""
            print(f"  ✓ {scenario.slug}{note}{warn}", flush=True)
            played[scenario.slug] = outcome
    finally:
        stack.restore()
    # Le rapport couvre toute la campagne : les scénarios déjà joués dans ce dossier sont réanalysés.
    outcomes = []
    for scenario in cfg.campaigns[args.campaign]:
        if scenario.slug in played:
            outcomes.append(played[scenario.slug])
        elif any((out / scenario.slug / "raw").glob("start-*.txt")):
            outcomes.append(analyse(scenario, cfg.slo, out / scenario.slug, prometheus))
    path = report(outcomes, cfg, out, args.campaign)
    print(f"\nRapport : {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
