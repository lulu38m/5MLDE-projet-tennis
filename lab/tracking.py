"""Traces des essais : expérience MLflow `lab-atp`, et un classement local (`state/lab/leaderboard.jsonl`).

Le MLflow de la stack (`make up`) est utilisé s'il répond ; sinon une base locale `state/lab/mlflow.db` le
remplace (`make lab-mlflow` l'ouvre). Le classement local marche dans tous les cas.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
import mlflow
import pandas as pd

from lab.evaluate import STATE, RunResult
from tennis_mlops.config import Settings

LEADERBOARD = STATE / "leaderboard.jsonl"
BEST = STATE / "best.json"
PREDICTIONS = STATE / "predictions"
LOCAL_URI = f"sqlite:///{(STATE / 'mlflow.db').as_posix()}"


def connect(experiment: str) -> str:
    """Branche MLflow (stack si joignable, sinon base locale) et renvoie l'URI utilisée."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    uri = str(Settings().mlflow_tracking_uri).rstrip("/")
    try:
        httpx.get(f"{uri}/health", timeout=2).raise_for_status()
    except httpx.HTTPError:
        STATE.mkdir(parents=True, exist_ok=True)
        print(f"MLflow de la stack injoignable ({uri}) : essais tracés dans {LOCAL_URI} (make lab-mlflow pour le voir)")
        uri = LOCAL_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
                              timeout=5).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "inconnu"


@contextmanager
def run(name: str, tags: dict[str, str], nested: bool = False) -> Iterator[Any]:
    """Un run MLflow, avec la révision Git et le jeu de features en tags."""
    with mlflow.start_run(run_name=name, nested=nested, tags={"git_commit": _git(), **tags}) as active:
        yield active


def log_result(result: RunResult, groups: dict[str, list[int]], extra: dict[str, Any], name: str) -> dict[str, Any]:
    """Métriques par saison et par groupe, paramètres, importance, prédictions ; ligne du classement local."""
    mlflow.log_params({f"model.{k}": v for k, v in result.params.items()} | {k: v for k, v in extra.items()
                      if not isinstance(v, dict | list)})  # fmt: skip
    for s in result.seasons:
        mlflow.log_metrics({f"s{s.season}.{m}": getattr(s, m) for m in ("log_loss", "accuracy", "brier", "auc")})
        if s.trees:
            mlflow.log_metric(f"s{s.season}.trees", s.trees)
    at = datetime.now(UTC).isoformat(timespec="seconds")
    summary: dict[str, Any] = {"name": name, "family": result.family, "at": at, "params": result.params, **extra}
    for group, seasons in groups.items():
        means = result.mean(seasons)
        mlflow.log_metrics({f"{group}.{m}": v for m, v in means.items()})
        summary |= {f"{group}.{m}": round(v, 5) for m, v in means.items()}
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    if not result.predictions.empty:
        path = PREDICTIONS / f"{name}.parquet"
        result.predictions.to_parquet(path)
        mlflow.log_artifact(str(path))
        summary["predictions"] = str(path)
    if result.importance is not None:
        imp = STATE / "importance.csv"
        result.importance.rename("importance").to_csv(imp)
        mlflow.log_artifact(str(imp))
    active = mlflow.active_run()
    summary["run_id"] = active.info.run_id if active is not None else None
    with LEADERBOARD.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, default=str, ensure_ascii=False) + "\n")
    _remember_best(summary)
    return summary


def _remember_best(summary: dict[str, Any]) -> None:
    """Meilleur essai de chaque famille (log loss d'évaluation) : ce que reprennent les ensembles."""
    if "eval.log_loss" not in summary or "predictions" not in summary or summary.get("holdout_only"):
        return
    best = json.loads(BEST.read_text(encoding="utf-8")) if BEST.exists() else {}
    current = best.get(summary["family"])
    if current is None or summary["eval.log_loss"] < current["eval.log_loss"]:
        best[summary["family"]] = summary
        BEST.write_text(json.dumps(best, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def best_runs() -> dict[str, dict[str, Any]]:
    """Le meilleur essai connu de chaque famille."""
    return json.loads(BEST.read_text(encoding="utf-8")) if BEST.exists() else {}


def leaderboard(top: int = 20, holdout: bool = False) -> pd.DataFrame:
    """Les essais, du meilleur au moins bon (log loss d'évaluation), avec l'accuracy à côté."""
    if not LEADERBOARD.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in LEADERBOARD.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if "eval.log_loss" not in frame:
        return frame
    columns = ["name", "family", "features", "eval.log_loss", "eval.accuracy", "tune.log_loss", "tune.accuracy",
               "eval.brier", "eval.auc"] + (["holdout.log_loss", "holdout.accuracy"] if holdout else [])  # fmt: skip
    frame = frame.dropna(subset=["eval.log_loss"]).sort_values("eval.log_loss")
    return frame[[c for c in columns if c in frame]].head(top).reset_index(drop=True)
