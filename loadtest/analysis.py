"""Ce qu'une campagne a mesuré : percentiles exacts par palier, chronologie, verdicts.

Tout part du journal des requêtes écrit par `locustfile.py` (une ligne par requête : heure, endpoint, durée,
succès) : les percentiles sont calculés sur toutes les requêtes du palier, après sa montée, et non sur les fenêtres
glissantes de Locust. Les requêtes en erreur comptent avec leur durée (un délai dépassé pèse sur le p95).

Un batch de 50 matchs n'est pas une requête comme les autres : les percentiles du palier (et l'objectif p95) portent
sur les requêtes unitaires ; les batchs ont leur propre p95 et leur propre objectif (`slo.batch_p95_ms`). Un
scénario fait uniquement de batchs est jugé sur ceux-ci.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from loadtest.profiles import Slo, Stage

COLUMNS = ["t", "name", "ms", "ok"]
BATCH = "/predict/batch"


def is_batch(requests: pd.DataFrame) -> pd.Series[bool]:
    """Les lignes qui sont des batchs."""
    return requests["name"].astype(str).str.startswith(BATCH)


def latency_basis(requests: pd.DataFrame) -> pd.DataFrame:
    """Les requêtes sur lesquelles on calcule les percentiles : les unitaires, ou les batchs s'il n'y a qu'eux."""
    single = requests[~is_batch(requests)]
    return single if not single.empty else requests


@dataclass(frozen=True)
class Window:
    """Un palier dans le temps (secondes epoch)."""

    stage: Stage
    start: float
    measure_from: float
    end: float


def load_requests(directory: Path) -> pd.DataFrame:
    """Les requêtes de tous les processus Locust, triées par heure."""
    frames = [pd.read_csv(p) for p in sorted(directory.glob("requests-*.csv")) if p.stat().st_size > 0]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS).astype({"t": float, "name": str, "ms": float, "ok": int})
    return pd.concat(frames, ignore_index=True).sort_values("t", ignore_index=True)


def start_time(directory: Path) -> float:
    """Début du test (le plus tôt des processus)."""
    marks = [float(p.read_text(encoding="utf-8").strip()) for p in directory.glob("start-*.txt")]
    if not marks:
        raise FileNotFoundError(f"aucun start-*.txt dans {directory} : Locust a-t-il démarré ?")
    return min(marks)


def windows(stages: list[Stage], t0: float) -> list[Window]:
    """Les paliers placés dans le temps à partir du début du test."""
    out, at = [], t0
    for stage in stages:
        out.append(Window(stage, at, at + stage.measured_from_s, at + stage.duration_s))
        at += stage.duration_s
    return out


def _summary(ms: npt.NDArray[np.float64], ok: npt.NDArray[np.float64], seconds: float) -> dict[str, float]:
    if len(ms) == 0:
        nan = float("nan")
        return {"requests": 0, "rps": 0.0, "error_rate": nan, "p50_ms": nan, "p95_ms": nan, "p99_ms": nan,
                "mean_ms": nan, "max_ms": nan}  # fmt: skip
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    return {"requests": len(ms), "rps": len(ms) / seconds if seconds > 0 else 0.0, "error_rate": 1 - float(ok.mean()),
            "p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99), "mean_ms": float(ms.mean()),
            "max_ms": float(ms.max())}  # fmt: skip


def stage_metrics(requests: pd.DataFrame, placed: list[Window], slo: Slo) -> pd.DataFrame:
    """Une ligne par palier : débit, erreurs, percentiles, et s'il tient l'objectif."""
    rows = []
    for i, w in enumerate(placed):
        inside = requests[(requests["t"] >= w.measure_from) & (requests["t"] < w.end)]
        seconds = w.end - w.measure_from
        everything = _summary(inside["ms"].to_numpy(dtype=float), inside["ok"].to_numpy(dtype=float), seconds)
        basis = latency_basis(inside)
        latency = _summary(basis["ms"].to_numpy(dtype=float), basis["ok"].to_numpy(dtype=float), seconds)
        batches = inside[is_batch(inside)]["ms"].to_numpy(dtype=float)
        batch_p95 = float(np.percentile(batches, 95)) if len(batches) else float("nan")
        only_batches = len(batches) == len(inside)
        held = bool(
            everything["requests"] > 0
            and everything["error_rate"] < slo.error_rate
            and (only_batches or latency["p95_ms"] < slo.p95_ms)
            and (len(batches) == 0 or batch_p95 < slo.batch_p95_ms)
        )
        rows.append({
            "stage": i + 1, "label": w.stage.label, "users": w.stage.users, "start": w.start,
            "measure_from": w.measure_from, "end": w.end, "requests": everything["requests"], "rps": everything["rps"],
            "ok_rps": everything["rps"] * (1 - (everything["error_rate"] if everything["requests"] else 0)),
            "error_rate": everything["error_rate"], "p50_ms": latency["p50_ms"], "p95_ms": latency["p95_ms"],
            "p99_ms": latency["p99_ms"], "mean_ms": latency["mean_ms"], "max_ms": latency["max_ms"],
            "batches": len(batches), "batch_p95_ms": batch_p95, "slo_ok": held,
        })  # fmt: skip
    return pd.DataFrame(rows)


def endpoint_metrics(requests: pd.DataFrame, placed: list[Window]) -> pd.DataFrame:
    """Percentiles par endpoint et par palier (où le temps passe-t-il ?)."""
    rows = []
    for i, w in enumerate(placed):
        inside = requests[(requests["t"] >= w.measure_from) & (requests["t"] < w.end)]
        for name, group in inside.groupby("name"):
            stats = _summary(group["ms"].to_numpy(dtype=float), group["ok"].to_numpy(dtype=float),
                             w.end - w.measure_from)  # fmt: skip
            rows.append({"stage": i + 1, "users": w.stage.users, "name": name, **stats})
    return pd.DataFrame(rows)


def timeline(requests: pd.DataFrame, t0: float, bucket_s: float = 10) -> pd.DataFrame:
    """Débit, erreurs et percentiles par tranche de `bucket_s` secondes depuis le début du test."""
    if requests.empty:
        return pd.DataFrame(columns=["second", "rps", "error_rate", "p50_ms", "p95_ms"])
    second = ((requests["t"] - t0) // bucket_s * bucket_s).rename("second")
    grouped = requests.groupby(second)
    basis = latency_basis(requests)
    latency = basis.groupby(second.loc[basis.index])["ms"]
    out = pd.DataFrame({
        "rps": grouped.size() / bucket_s,
        "error_rate": 1 - grouped["ok"].mean(),
        "p50_ms": latency.quantile(0.5),
        "p95_ms": latency.quantile(0.95),
    })  # fmt: skip
    return out.reset_index()


def longest_pause(requests: pd.DataFrame, t0: float, end: float) -> tuple[float, float]:
    """Le plus long moment sans aucune requête (s), et quand il commence (s depuis le début).

    Les utilisateurs envoient au moins une requête toutes les 1,5 s : un trou de plusieurs secondes veut dire que
    Locust s'est arrêté (machine en veille, processus figé), et le scénario est à relancer.
    """
    times = np.concatenate([[t0], requests["t"].to_numpy(dtype=float), [end]])
    gaps = np.diff(times)
    i = int(np.argmax(gaps))
    return float(gaps[i]), float(times[i] - t0)


@dataclass(frozen=True)
class Capacity:
    """Le dernier palier tenu d'une rampe (tous les précédents l'étant aussi)."""

    users: int
    rps: float
    p95_ms: float
    first_failure: int | None  # utilisateurs du premier palier qui ne tient plus


def capacity(stages: pd.DataFrame) -> Capacity | None:
    """Le nombre maximal d'utilisateurs simultanés qui tient l'objectif ; None si le premier palier échoue déjà."""
    best = None
    for row in stages.to_dict("records"):
        if not row["slo_ok"]:
            failure = int(row["users"])
            return Capacity(best.users, best.rps, best.p95_ms, failure) if best else None
        best = Capacity(int(row["users"]), float(row["rps"]), float(row["p95_ms"]), None)
    return best


@dataclass(frozen=True)
class SpikeResult:
    """Ce qu'un pic fait au service."""

    base_p95_ms: float
    peak_p95_ms: float
    peak_error_rate: float
    after_p95_ms: float
    recovery_s: float | None  # depuis la fin du pic, jusqu'au retour durable sous l'objectif ; None : jamais


def spike(stages: pd.DataFrame, line: pd.DataFrame, placed: list[Window], t0: float, slo: Slo) -> SpikeResult:
    """Dégradation pendant le pic et temps de retour à la normale."""
    before, during, after = stages.iloc[0], stages.iloc[1], stages.iloc[2]
    end_of_peak = placed[1].end - t0
    later = line[line["second"] >= end_of_peak].reset_index(drop=True)
    good = (later["p95_ms"] < slo.p95_ms) & (later["error_rate"] < slo.error_rate)
    recovery = None
    for i in range(len(later)):
        if good.iloc[i:].all():  # bon jusqu'à la fin du test
            recovery = float(later["second"].iloc[i] - end_of_peak)
            break
    return SpikeResult(float(before["p95_ms"]), float(during["p95_ms"]), float(during["error_rate"]),
                       float(after["p95_ms"]), recovery)  # fmt: skip


@dataclass(frozen=True)
class LeakResult:
    """Mémoire de l'API pendant l'endurance."""

    start_mb: float
    end_mb: float
    slope_mb_per_10min: float
    suspected: bool


def leak(memory: pd.DataFrame, skip_s: float = 120, threshold_mb_per_10min: float = 20) -> LeakResult | None:
    """Pente de la mémoire (colonnes `second`, `bytes`) après la mise en route ; fuite si elle monte franchement."""
    data = memory[memory["second"] >= skip_s]
    if len(data) < 3:
        return None
    mb = data["bytes"].to_numpy(dtype=float) / 1e6
    slope = float(np.polyfit(data["second"].to_numpy(dtype=float), mb, 1)[0]) * 600
    return LeakResult(float(mb[0]), float(mb[-1]), slope, slope > threshold_mb_per_10min)
