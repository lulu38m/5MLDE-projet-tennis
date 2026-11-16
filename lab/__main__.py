"""`python -m lab <commande>` : le labo d'entraînement (voir `lab/README.md`).

features [--refresh]            calcule (ou recalcule) les features de tous les matchs, mises en cache
reference                       la référence à battre : features et modèle du pipeline, mêmes saisons
run --model M [--features S]    un essai : saisons tune + eval (+ holdout avec --holdout), tracé dans MLflow
tune --model M --trials N       réglage Optuna sur les saisons tune, puis l'essai gagnant sur eval
ensemble [--members a,b,c]      moyenne (ou combinaison apprise, --fit) des meilleurs essais de chaque famille
leaderboard [--top N]           le classement des essais (log loss et accuracy)
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from lab import tracking
from lab.evaluate import RunResult, SeasonResult, feature_frame, load_config, metrics, select, walk_forward
from lab.models import logit, sigmoid


def _groups(config: dict[str, Any], holdout: bool) -> dict[str, list[int]]:
    groups = {"tune": list(config["seasons"]["tune"]), "eval": list(config["seasons"]["eval"])}
    if holdout:
        groups["holdout"] = list(config["seasons"]["holdout"])
    return groups


def _print(summary: dict[str, Any], holdout: bool) -> None:
    def fmt(group: str) -> str:
        ll, acc = summary.get(f"{group}.log_loss"), summary.get(f"{group}.accuracy")
        if ll is None or acc is None:
            return f"{group} : —"
        return f"{group} : log loss {ll:.4f} · accuracy {acc * 100:.2f} %"

    lines = [fmt("tune"), fmt("eval")] + ([fmt("holdout")] if holdout else [])
    print(f"\n{summary['name']}\n  " + "\n  ".join(lines))


def cmd_features(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Commande `features`."""
    frame, key = feature_frame(config, refresh=args.refresh)
    print(f"{len(frame)} matchs, du {frame['match_date'].min()} au {frame['match_date'].max()} (cache {key})")


def cmd_reference(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Commande `reference`."""
    from lab.reference import pipeline_reference

    groups = _groups(config, args.holdout)
    tracking.connect(config["experiment"])
    result = pipeline_reference(sorted({s for g in groups.values() for s in g}), config)
    with tracking.run("référence pipeline (LR C=0,05)", {"kind": "reference"}):
        summary = tracking.log_result(result, groups, {"features": "pipeline"}, f"reference-{int(time.time())}")
    _print(summary, args.holdout)


def run_once(config: dict[str, Any], family: str, params: dict[str, Any], feature_set: str, holdout: bool,
             label: str, recency: float | None = None, nested: bool = False) -> dict[str, Any]:  # fmt: skip
    """Un essai complet, tracé."""
    frame, key = feature_frame(config)
    columns = select(frame, config["feature_sets"][feature_set])
    cfg = config if recency is None else config | {"recency_half_life_years": recency}
    groups = _groups(config, holdout)
    started = time.perf_counter()
    result = walk_forward(frame, family, params, columns, sorted({s for g in groups.values() for s in g}), cfg)
    name = f"{family}-{feature_set}-{label}-{int(time.time())}"
    tags = {"kind": label, "family": family, "features": feature_set, "features_cache": key}
    with tracking.run(name, tags, nested=nested):
        extra = {"features": feature_set, "columns": len(columns), "recency_half_life_years":
                 cfg.get("recency_half_life_years", 0), "seconds": round(time.perf_counter() - started, 1)}  # fmt: skip
        summary = tracking.log_result(result, groups, extra, name)
    return summary


def cmd_run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Commande `run`."""
    tracking.connect(config["experiment"])
    params = dict(config["models"].get(args.model, {})) | json.loads(args.params or "{}")
    _print(run_once(config, args.model, params, args.features, args.holdout, "run"), args.holdout)


def cmd_tune(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Commande `tune`."""
    import mlflow

    from lab.tune import split_best, study

    tracking.connect(config["experiment"])
    frame, _ = feature_frame(config)
    with tracking.run(f"tune-{args.model}-{int(time.time())}", {"kind": "tune", "family": args.model}):

        def on_trial(trial: Any, params: dict[str, Any], means: dict[str, float]) -> None:  # noqa: ANN401
            with tracking.run(f"essai {trial.number}", {"kind": "trial"}, nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({f"tune.{k}": v for k, v in means.items()})
            ll, acc = means["log_loss"], means["accuracy"] * 100
            print(f"  essai {trial.number:3d} : log loss {ll:.4f} · accuracy {acc:.2f} %")

        found = study(frame, args.model, config, args.trials, args.timeout, on_trial)
        complete = [t for t in found.trials if t.state.name == "COMPLETE"]
        print(f"\n{len(complete)} essais terminés, {len(found.trials) - len(complete)} arrêtés en route ; meilleur : "
              f"{found.best_value:.4f} "
              f"(essai {found.best_trial.number}) {found.best_params}")  # fmt: skip
        mlflow.log_params({f"best.{k}": v for k, v in found.best_params.items()})
        mlflow.log_metric("best.tune.log_loss", found.best_value)
        model_params, feature_set, recency = split_best(found.best_params)
        summary = run_once(config, args.model, model_params, feature_set, args.holdout, "tuned", recency, nested=True)
    _print(summary, args.holdout)


def cmd_ensemble(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Commande `ensemble`."""
    import mlflow

    best = tracking.best_runs()
    members = args.members.split(",") if args.members else [f for f in best if f not in {"elo", "ensemble"}]
    missing = [m for m in members if m not in best]
    if missing:
        msg = f"pas encore d'essai pour {', '.join(missing)} : lancer d'abord `make lab-run MODEL=…` ou `make lab-tune`"
        raise SystemExit(msg)
    paths = getattr(args, "predictions", None) or {m: best[m]["predictions"] for m in members}
    frames = [pd.read_parquet(paths[m]).set_index(["match_id", "season"]) for m in members]
    joined = pd.concat([f["p"].rename(m) for f, m in zip(frames, members, strict=True)], axis=1, join="inner")
    labels = frames[0]["label"].reindex(joined.index)
    groups = _groups(config, args.holdout)
    z = logit(joined.to_numpy())
    weights = np.full(len(members), 1 / len(members))
    if args.fit:  # combinaison apprise sur les saisons tune, jugée sur les saisons eval
        from sklearn.linear_model import LogisticRegression

        tune_rows = joined.index.get_level_values("season").isin(groups["tune"])
        stack = LogisticRegression(C=1.0).fit(z[tune_rows], labels[tune_rows])
        weights = stack.coef_[0]
        p = stack.predict_proba(z)[:, 1]
    else:
        p = sigmoid(z @ weights)
    result = RunResult("ensemble", {"members": ",".join(members), "fit": args.fit,
                                    "weights": ",".join(f"{w:.3f}" for w in weights)})  # fmt: skip
    seasons = joined.index.get_level_values("season")
    for season in sorted(set(seasons)):
        rows = seasons == season
        m = metrics(labels[rows].to_numpy(), p[rows])
        result.seasons.append(SeasonResult.of(int(season), int(rows.sum()), m, 0.0))
    result.predictions = pd.DataFrame({"match_id": joined.index.get_level_values("match_id"), "season": seasons,
                                       "label": labels.to_numpy(), "p": p})  # fmt: skip
    tracking.connect(config["experiment"])
    name = f"ensemble-{'fit' if args.fit else 'moyenne'}-{int(time.time())}"
    with tracking.run(name, {"kind": "ensemble", "family": "ensemble", "features": "membres"}):
        summary = tracking.log_result(result, groups, {"features": "+".join(members)}, name)
        mlflow.log_param("weights", ",".join(f"{w:.3f}" for w in weights))
    _print(summary, args.holdout)
    print(f"  poids : {dict(zip(members, [round(float(w), 3) for w in weights], strict=True))}")


def cmd_final(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Commande `final` : rejoue le meilleur essai de chaque famille avec la saison holdout, puis l'ensemble.

    À ne lancer qu'une fois les choix arrêtés : regarder le holdout pour choisir, c'est le transformer en eval.
    """
    tracking.connect(config["experiment"])
    best = {f: b for f, b in tracking.best_runs().items() if f not in {"elo", "ensemble"}}
    if not best:
        raise SystemExit("aucun essai à rejouer : make lab-run ou make lab-tune d'abord")
    predictions = {}
    for family, entry in best.items():
        summary = run_once(config, family, dict(entry["params"]), str(entry["features"]), True, "final",
                           float(entry.get("recency_half_life_years") or 0))  # fmt: skip
        predictions[family] = summary["predictions"]
        _print(summary, True)
    cmd_ensemble(argparse.Namespace(members=",".join(best), fit=True, holdout=True, predictions=predictions), config)


def cmd_leaderboard(args: argparse.Namespace, _: dict[str, Any]) -> None:
    """Commande `leaderboard`."""
    board = tracking.leaderboard(args.top, args.holdout)
    if board.empty:
        print("aucun essai pour l'instant : make lab-reference, puis make lab-run MODEL=lightgbm")
        return
    with pd.option_context("display.width", 200, "display.max_colwidth", 48):
        print(board.to_string(float_format=lambda v: f"{v:.4f}"))


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*(eval_set|iprint).*")
    parser = argparse.ArgumentParser(prog="python -m lab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("features")
    p.add_argument("--refresh", action="store_true")
    p = sub.add_parser("reference")
    p.add_argument("--holdout", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--model", required=True)
    p.add_argument("--features", default="all")
    p.add_argument("--params", help='JSON, ex. {"num_leaves": 15}')
    p.add_argument("--holdout", action="store_true")
    p = sub.add_parser("tune")
    p.add_argument("--model", required=True)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--timeout", type=float, help="secondes")
    p.add_argument("--holdout", action="store_true")
    p = sub.add_parser("ensemble")
    p.add_argument("--members", help="familles séparées par des virgules (défaut : toutes celles déjà essayées)")
    p.add_argument("--fit", action="store_true", help="poids appris sur les saisons tune")
    p.add_argument("--holdout", action="store_true")
    sub.add_parser("final")
    p = sub.add_parser("leaderboard")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--holdout", action="store_true")
    args = parser.parse_args(argv)
    commands = {"features": cmd_features, "reference": cmd_reference, "run": cmd_run, "tune": cmd_tune,
                "ensemble": cmd_ensemble, "leaderboard": cmd_leaderboard, "final": cmd_final}  # fmt: skip
    commands[args.command](args, load_config())


if __name__ == "__main__":
    main()
