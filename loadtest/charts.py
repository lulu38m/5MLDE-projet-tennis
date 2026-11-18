"""Graphiques de la campagne (PNG, matplotlib sans affichage)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from loadtest.profiles import Slo

COLORS = {"A": "#d1495b", "B": "#2e86ab", "C": "#3bb273", "p50": "#8fb8de", "p95": "#2e86ab", "p99": "#1b3b6f"}


def _slo_line(ax: Axes, slo: Slo) -> None:
    ax.axhline(slo.p95_ms, color="#d1495b", linestyle="--", linewidth=1, label=f"objectif p95 {slo.p95_ms:.0f} ms")


def ramp(stages: pd.DataFrame, title: str, slo: Slo, path: Path) -> None:
    """Latences et débit selon le nombre d'utilisateurs."""
    fig, (lat, thr) = plt.subplots(1, 2, figsize=(12, 4.5))
    for q in ("p50", "p95", "p99"):
        lat.plot(stages["users"], stages[f"{q}_ms"], marker="o", color=COLORS[q], label=q)
    _slo_line(lat, slo)
    lat.set(xscale="log", yscale="log", xlabel="utilisateurs simultanés", ylabel="latence (ms)", title="Latence")
    lat.legend()
    thr.plot(stages["users"], stages["ok_rps"], marker="o", color=COLORS["p95"], label="réponses réussies/s")
    errors = thr.twinx()
    errors.bar(stages["users"], stages["error_rate"] * 100, width=stages["users"] * 0.25, alpha=0.3,
               color="#d1495b", label="erreurs (%)")  # fmt: skip
    thr.set(
        xscale="log", xlabel="utilisateurs simultanés", ylabel="réponses réussies/s", title="Débit utile et erreurs"
    )
    errors.set_ylim(0, max(5.0, float(stages["error_rate"].max() * 100) * 1.2))
    errors.set_ylabel("erreurs (%)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def compare(ramps: dict[str, pd.DataFrame], slo: Slo, path: Path) -> None:
    """p95 et débit de chaque configuration, palier par palier."""
    fig, (lat, thr) = plt.subplots(1, 2, figsize=(12, 4.5))
    for config, stages in ramps.items():
        color = COLORS.get(config)
        lat.plot(stages["users"], stages["p95_ms"], marker="o", color=color, label=f"config {config}")
        thr.plot(stages["users"], stages["ok_rps"], marker="o", color=color, label=f"config {config}")
    _slo_line(lat, slo)
    lat.set(xscale="log", yscale="log", xlabel="utilisateurs simultanés", ylabel="p95 (ms)", title="p95 par config")
    thr.set(
        xscale="log", xlabel="utilisateurs simultanés", ylabel="réponses réussies/s", title="Débit utile par config"
    )
    lat.legend()
    thr.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def timeline(line: pd.DataFrame, users: pd.DataFrame, title: str, slo: Slo, path: Path,
             memory: pd.DataFrame | None = None) -> None:  # fmt: skip
    """Chronologie : p95, débit, utilisateurs visés (et mémoire de l'API pour l'endurance)."""
    rows = 3 if memory is not None and not memory.empty else 2
    fig, axes = plt.subplots(rows, 1, figsize=(11, 2.6 * rows), sharex=True)
    lat, thr = axes[0], axes[1]
    lat.plot(line["second"], line["p95_ms"], color=COLORS["p95"], label="p95")
    lat.plot(line["second"], line["p50_ms"], color=COLORS["p50"], label="p50")
    _slo_line(lat, slo)
    lat.set(yscale="log", ylabel="latence (ms)")
    lat.legend(loc="upper left")
    thr.plot(line["second"], line["rps"], color="#3bb273", label="requêtes/s")
    thr.set_ylabel("requêtes/s")
    target = thr.twinx()
    end = pd.DataFrame({"second": [line["second"].max() + 10], "users": [users["users"].iloc[-1]]})
    steps = pd.concat([users, end], ignore_index=True)
    target.step(steps["second"], steps["users"], where="post", color="#555555", linestyle=":", label="utilisateurs")
    target.set_ylabel("utilisateurs")
    if rows == 3 and memory is not None:
        axes[2].plot(memory["second"], memory["bytes"] / 1e6, color="#7b2d26")
        axes[2].set_ylabel("mémoire API (Mo)")
    axes[-1].set_xlabel("secondes depuis le début")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
