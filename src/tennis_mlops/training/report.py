"""Journalisation d'un candidat et de la campagne : paramètres, métriques, graphiques, fiche modèle, modèle.

Les graphiques utilisent l'API objet de matplotlib (`Figure`), sans `pyplot` : elle n'a pas d'état global et peut
servir depuis plusieurs threads (les candidats sont journalisés en parallèle).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import skops.io
from matplotlib.figure import Figure
from mlflow.models import ModelSignature, infer_signature
from mlflow.models.model import ModelInfo
from sklearn.metrics import confusion_matrix, roc_curve

from tennis_mlops.features import FEATURE_COLUMNS, FEATURES_VERSION
from tennis_mlops.registry.events import RegistryEvent
from tennis_mlops.registry.gate import GateDecision
from tennis_mlops.registry.model_registry import RegisteredVersion
from tennis_mlops.training import html_report as h
from tennis_mlops.training.config import TrainConfig
from tennis_mlops.training.dataset import Dataset
from tennis_mlops.training.evaluate import Array, Evaluation, ProbabilityModel, calibration_table
from tennis_mlops.training.split import Split
from tennis_mlops.training.tracking import MlflowTracker
from tennis_mlops.training.trainer import FittedCandidate

#: Seuls des types de scikit-learn (et le `dtype` de NumPy) peuvent être déclarés de confiance au chargement skops.
TRUSTED_PREFIXES = ("sklearn.",)
TRUSTED_EXACT = frozenset({"numpy.dtype"})
_METRIC_KEY = re.compile(r"[^\w.\- /]")


class UntrustedTypeError(RuntimeError):
    """Le modèle contient un type qui ne vient pas de scikit-learn : on refuse de le déclarer de confiance."""


def trusted_types(model: object) -> list[str]:
    """Types à déclarer au chargement skops ; refuse tout type étranger à scikit-learn."""
    untrusted: list[str] = skops.io.get_untrusted_types(data=skops.io.dumps(model))
    foreign = [t for t in untrusted if t not in TRUSTED_EXACT and not t.startswith(TRUSTED_PREFIXES)]
    if foreign:
        msg = f"types non issus de scikit-learn dans le modèle : {', '.join(foreign)}"
        raise UntrustedTypeError(msg)
    return sorted(untrusted)


def model_contract(model: ProbabilityModel, features: pd.DataFrame) -> tuple[ModelSignature, pd.DataFrame]:
    """Signature (entrée = features brutes, sortie = probabilités des deux classes) et petit exemple d'entrée."""
    sample = features.head(200)
    return infer_signature(sample, model.predict_proba(sample)), features.head(5)


def metric_key(section: str, *parts: object) -> str:
    """Nom de métrique accepté par MLflow, dans une section : `segments/ecart_classement.200plus`."""
    cleaned = (str(p).replace("+", "plus").replace("/", "-") for p in parts)
    return f"{section}/" + _METRIC_KEY.sub("_", ".".join(cleaned))


# --------------------------------------------------------------------------- graphiques


def _figure(width: float = 6.0, height: float = 4.2) -> tuple[Figure, Any]:
    figure = Figure(figsize=(width, height))
    return figure, figure.subplots()


def calibration_figure(labels: Array, probabilities: Array, bins: int, title: str) -> Figure:
    """Probabilité annoncée contre fréquence observée ; la diagonale est la calibration parfaite."""
    table = calibration_table(labels, probabilities, bins)
    figure, ax = _figure()
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="calibration parfaite")
    ax.plot(table["mean_probability"], table["observed_rate"], marker="o", label="modèle")
    ax.set(
        xlabel="probabilité annoncée (joueur 1 gagne)",
        ylabel="taux de victoire observé",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    return figure


def roc_figure(labels: Array, probabilities: Array, auc: float, title: str) -> Figure:
    """Courbe ROC."""
    fpr, tpr, _ = roc_curve(labels, probabilities)
    figure, ax = _figure()
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set(xlabel="taux de faux positifs", ylabel="taux de vrais positifs", title=title)
    ax.legend(loc="lower right")
    return figure


def confusion_figure(labels: Array, probabilities: Array, title: str) -> Figure:
    """Matrice de confusion au seuil 0,5."""
    matrix = confusion_matrix(labels, probabilities >= 0.5, labels=[0, 1])
    figure, ax = _figure(4.6, 4.0)
    ax.imshow(matrix, cmap="Blues")
    names = ["joueur 2 gagne", "joueur 1 gagne"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    ax.set(xlabel="prédit", ylabel="réel", title=title)
    return figure


def distribution_figure(labels: Array, probabilities: Array, title: str) -> Figure:
    """Distribution des probabilités, séparée selon le vrai vainqueur."""
    figure, ax = _figure()
    bins = np.linspace(0, 1, 21)
    ax.hist(probabilities[labels == 1], bins=bins, alpha=0.6, label="le joueur 1 a gagné")
    ax.hist(probabilities[labels == 0], bins=bins, alpha=0.6, label="le joueur 2 a gagné")
    ax.set(xlabel="probabilité que le joueur 1 gagne", ylabel="matchs", title=title)
    ax.legend()
    return figure


def importance_figure(importance: pd.DataFrame, title: str) -> Figure:
    """Importance par permutation (hausse du log loss), de la plus forte à la plus faible."""
    ordered = importance.sort_values("importance")
    figure, ax = _figure(6.0, 0.28 * len(ordered) + 1.2)
    ax.barh(ordered["feature"], ordered["importance"], xerr=ordered["std"], color="#4C72B0")
    ax.set(xlabel="hausse du log loss quand la feature est mélangée", title=title)
    return figure


def segments_figure(segments: pd.DataFrame, overall: float, title: str) -> Figure:
    """Log loss par segment (test), comparé au log loss global."""
    data = segments.dropna(subset=["log_loss"]) if "log_loss" in segments else segments.iloc[0:0]
    labels = [f"{r.dimension} · {r.segment} ({r.rows})" for r in data.itertuples()]
    figure, ax = _figure(6.4, 0.3 * max(len(labels), 1) + 1.4)
    ax.barh(labels[::-1], data["log_loss"].to_numpy()[::-1], color="#55A868")
    ax.axvline(overall, color="grey", linestyle="--", linewidth=1, label=f"global = {overall:.3f}")
    ax.set(xlabel="log loss (test, plus bas = mieux)", title=title)
    ax.legend(loc="lower right")
    return figure


def candidates_figure(table: pd.DataFrame, baselines: Mapping[str, float], metric: str) -> Figure:
    """Comparaison des candidats sur la validation, avec les baselines en pointillés."""
    column = f"validation_{metric}"
    ordered = table.sort_values(column, ascending=False)
    figure, ax = _figure(6.6, 0.42 * len(ordered) + 1.6)
    colors = ["#C44E52" if best else "#4C72B0" for best in ordered["selected"]]
    ax.barh(ordered["candidate"], ordered[column], color=colors)
    for (name, value), style in zip(baselines.items(), ("--", ":"), strict=False):
        ax.axvline(value, color="grey", linestyle=style, linewidth=1, label=f"baseline {name} = {value:.4f}")
    low = min([*ordered[column], *baselines.values()])
    high = max([*ordered[column], *baselines.values()])
    ax.set_xlim(low - (high - low) * 0.5 - 1e-3, high + (high - low) * 0.2 + 1e-3)
    ax.set(
        xlabel=f"{metric} (validation, plus bas = mieux) — en rouge : le candidat retenu",
        title="Candidats de la campagne",
    )
    ax.legend(loc="lower right")
    return figure


def monthly_figure(monthly: pd.DataFrame, title: str) -> Figure:
    """Log loss du modèle et de l'Elo pur, mois par mois (validation puis test)."""
    figure, ax = _figure(6.8, 3.8)
    x = range(len(monthly))
    ax.plot(x, monthly["log_loss"], marker="o", label="modèle")
    ax.plot(x, monthly["reference_log_loss"], marker="o", linestyle="--", color="grey", label="Elo pur")
    test_start = next((i for i, part in enumerate(monthly["period"]) if part == "test"), None)
    if test_start is not None:
        ax.axvspan(test_start - 0.5, len(monthly) - 0.5, color="#4C72B0", alpha=0.08, label="période de test")
    ax.set_xticks(list(x), list(monthly["month"]), rotation=45, ha="right")
    ax.set(ylabel="log loss (plus bas = mieux)", title=title)
    ax.legend()
    return figure


def learning_figure(curve: pd.DataFrame, title: str) -> Figure:
    """Courbe d'apprentissage : l'écart entre les deux courbes qui grandit signale le surapprentissage."""
    figure, ax = _figure(6.0, 3.8)
    ax.plot(curve["iteration"], curve["train_log_loss"], label="entraînement")
    ax.plot(curve["iteration"], curve["validation_log_loss"], label="validation")
    best = int(curve["validation_log_loss"].idxmin())
    ax.axvline(
        curve["iteration"].iloc[best],
        color="grey",
        linestyle=":",
        linewidth=1,
        label=f"meilleure itération en validation : {curve['iteration'].iloc[best]}",
    )
    ax.set(xlabel="itération", ylabel="log loss", title=title)
    ax.legend()
    return figure


# --------------------------------------------------------------------------- fiche modèle


def model_card(
    fitted: FittedCandidate,
    evaluation: Evaluation,
    baselines: Mapping[str, Evaluation],
    split: Split,
    dataset: Dataset,
    importance: pd.DataFrame,
    config: TrainConfig,
) -> str:
    """Fiche Markdown : usage prévu, données, métriques, limites."""
    test = evaluation.metrics["test"]
    lines = [
        f"# Fiche modèle — {fitted.name}",
        "",
        f"Générée le {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Famille `{fitted.spec.kind.value}`, "
        f"hyperparamètres `{dict(fitted.spec.params)}`, graine {fitted.spec.seed}.",
        "",
        "## Usage prévu",
        "",
        "Probabilité que le **joueur 1** gagne un match ATP en simple, à partir d'informations connues avant le "
        "match (classement publié, Elo, forme, face-à-face). Usage pédagogique (projet 5MLDE) : ce modèle n'est "
        "pas conçu pour les paris.",
        "",
        "## Données",
        "",
        f"- Matchs validés par Great Expectations, `dataset_version` {', '.join(dataset.dataset_versions) or '?'} ; "
        f"{dataset.lots.get('lots_valid', '?')} lots valides, dernier lot `{dataset.lots.get('last_valid_lot')}`.",
        f"- Entraînement {split.train_start} → {split.validation_start} ({len(split.train)} matchs), "
        f"validation → {split.test_start} ({len(split.validation)}), test → {split.end} ({len(split.test)}).",
        f"- Features version {FEATURES_VERSION} : {', '.join(FEATURE_COLUMNS)}.",
        "",
        "## Métriques (test, jamais utilisé pour choisir)",
        "",
        "| Modèle | log loss | Brier | accuracy | ROC AUC | ECE |",
        "|---|---:|---:|---:|---:|---:|",
        _card_row(fitted.name, test),
        *(_card_row(f"baseline {name}", ev.metrics["test"]) for name, ev in baselines.items()),
        "",
        f"Sélection sur la validation ({config.selection_metric}). "
        f"Features les plus utiles : {', '.join(importance['feature'].head(5))}.",
        "",
        "## Limites",
        "",
        "- Pas d'information sur les blessures, la fatigue intra-tournoi ou les conditions de jeu.",
        "- Un joueur sans historique part d'un Elo moyen : ses premières prédictions sont peu informées.",
        "- Les segments à faible effectif (Carpet, Coupe Davis…) ont des métriques instables "
        "(voir `evaluation/segments.csv`).",
        "- Entraîné sur une fenêtre passée : la performance doit être surveillée sur les nouveaux matchs (dérive).",
    ]
    return "\n".join(lines) + "\n"


def _card_row(name: str, metrics: Mapping[str, float]) -> str:
    keys = ("log_loss", "brier", "accuracy", "roc_auc", "ece")
    return f"| {name} | " + " | ".join(f"{metrics[k]:.4f}" for k in keys) + " |"


# --------------------------------------------------------------------------- rapports HTML


def candidate_html(
    fitted: FittedCandidate,
    evaluation: Evaluation,
    baselines: Mapping[str, Evaluation],
    segment_table: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    split: Split,
    figures: Mapping[str, Figure],
    card: str,
) -> str:
    """`rapport.html` d'un candidat : chiffres clés, scores contre les baselines, temps, graphiques, erreurs, fiche."""
    test = evaluation.metrics["test"]
    elo = baselines["elo"].metrics["test"] if "elo" in baselines else test
    costs: Mapping[str, float] = diagnostics["costs"]
    key_figures = [
        h.Card(
            "log loss (test)",
            f"{test['log_loss']:.4f}",
            f"{test['log_loss'] - elo['log_loss']:+.4f} contre l'Elo pur",
            test["log_loss"] < elo["log_loss"],
        ),
        h.Card(
            "accuracy (test)",
            f"{test['accuracy']:.1%}",
            f"{(test['accuracy'] - elo['accuracy']) * 100:+.1f} points contre l'Elo pur",
            test["accuracy"] > elo["accuracy"],
        ),
        h.Card(
            "ROC AUC (test)",
            f"{test['roc_auc']:.3f}",
            f"{test['roc_auc'] - elo['roc_auc']:+.3f} contre l'Elo pur",
            test["roc_auc"] > elo["roc_auc"],
        ),
        h.Card("calibration ECE (test)", f"{test['ece']:.3f}", "0 = probabilités parfaitement calibrées"),
        h.Card("entraînement", f"{fitted.fit_seconds:.1f} s", f"{int(test['rows'])} matchs de test"),
        h.Card("latence d'une ligne", f"{costs['cost/latency_1_row_p95_ms']:.1f} ms", "p95"),
    ]
    scores = pd.DataFrame(
        [{"modèle": fitted.name, "partition": part, **_rounded(evaluation.metrics[part])} for part in PART_ORDER]
        + [
            {"modèle": f"baseline {name}", "partition": part, **_rounded(ev.metrics[part])}
            for name, ev in baselines.items()
            for part in ("validation", "test")
        ]
    )
    monthly: pd.DataFrame = diagnostics["monthly"]
    curve: pd.DataFrame | None = diagnostics.get("learning")
    plots = {name: fig for name, fig in figures.items() if name != "monthly"}
    body = [
        h.cards(key_figures),
        h.section(
            "Scores",
            h.paragraph(
                "Le choix entre candidats se fait sur la <strong>validation</strong> ; le "
                "<strong>test</strong> n'a servi qu'au bilan."
            ),
            h.table(scores),
        ),
        h.section(
            "Dans le temps",
            h.paragraph(
                "Un mois au-dessus de la ligne de l'Elo pur est un mois où le modèle fait "
                "moins bien que la baseline sans apprentissage."
            ),
            f'<img class="wide" alt="mois par mois" src="{h.figure_src(figures["monthly"])}">',
            h.table(monthly.drop(columns=["month_start"])),
        ),
        h.section("Graphiques", h.images(plots)),
        h.section("Par segment (test)", h.table(segment_table, scroll=True)),
        h.section(
            "Les plus grosses surprises (test)",
            h.paragraph("Matchs où le modèle était le plus sûr de lui et s'est trompé."),
            h.table(diagnostics["worst"].drop(columns=["match_id"]), digits=3, scroll=True),
        ),
        h.section(
            "Coûts",
            h.table(
                pd.DataFrame({"mesure": ["cost/fit_s", *costs], "valeur": [fitted.fit_seconds, *costs.values()]}),
                digits=2,
            ),
        ),
    ]
    if curve is not None:
        best = curve.loc[curve["validation_log_loss"].idxmin()]
        body.append(
            h.section(
                "Apprentissage",
                h.paragraph(
                    f"Meilleure itération en validation : {int(best['iteration'])} sur {len(curve)} "
                    f"(log loss {best['validation_log_loss']:.4f}). Au-delà, l'écart entre entraînement et "
                    "validation grandit : le modèle apprend le bruit de l'entraînement."
                ),
            )
        )
    body.append(h.section("Fiche modèle", h.markdown(card)))
    lead = (
        f"Famille <code>{h.esc(fitted.spec.kind.value)}</code> · "
        f"hyperparamètres <code>{h.esc(dict(fitted.spec.params))}</code> · "
        f"graine {fitted.spec.seed} · entraînement {split.train_start} → {split.validation_start}, validation → "
        f"{split.test_start}, test → {split.end}"
    )
    return h.page(f"Rapport — {fitted.name}", lead, body)


def campaign_html(summary: CampaignSummary, table: pd.DataFrame, figures: Mapping[str, Figure], metric: str) -> str:
    """`rapport_campagne.html` : le retenu, les candidats, les données, la dérive, les durées, la reproductibilité."""
    best, ctx, dataset, split = summary.best, summary.context, summary.dataset, summary.split
    elo = summary.baselines["elo"].metrics if "elo" in summary.baselines else {"test": best.test}
    key_figures = [
        h.Card(
            "candidat retenu", best.name, f"{_count(len(summary.results), 'candidat')}, {len(summary.failed)} en échec"
        ),
        h.Card(f"{metric} (validation)", f"{best.validation[metric]:.4f}", "sert à choisir"),
        h.Card(
            f"{metric} (test)",
            f"{best.test[metric]:.4f}",
            f"{best.test[metric] - elo['test'][metric]:+.4f} contre l'Elo pur",
            best.test[metric] < elo["test"][metric],
        ),
        h.Card(
            "registre",
            f"{summary.registered.name} v{summary.registered.version}",
            _decision_label(summary),
            None if summary.decision is None else summary.decision.passed,
        ),
        h.Card(
            "entraînement",
            f"{summary.durations['duration/training_wall_s']:.1f} s",
            f"{summary.workers} en parallèle, {summary.fit_sum:.1f} s cumulés",
        ),
    ]
    baselines = pd.DataFrame(
        [
            {"baseline": name, "partition": part, **_rounded(ev.metrics[part])}
            for name, ev in summary.baselines.items()
            for part in ("validation", "test")
        ]
    )
    data = pd.DataFrame(
        {
            "élément": [
                "matchs en base",
                "matchs joués (lignes)",
                "période",
                "dataset_version",
                "lots valides / en quarantaine",
                "dernier lot validé",
                "watermark",
                "entraînement",
                "validation",
                "test",
            ],
            "valeur": [
                dataset.matches_loaded,
                len(dataset.frame),
                f"{dataset.first_date} → {dataset.last_date}",
                ", ".join(dataset.dataset_versions),
                f"{dataset.lots.get('lots_valid')} / {dataset.lots.get('lots_quarantined')}",
                dataset.lots.get("last_valid_lot"),
                dataset.lots.get("watermark"),
                f"{split.train_start} → {split.validation_start} ({len(split.train)} matchs)",
                f"→ {split.test_start} ({len(split.validation)} matchs)",
                f"→ {split.end} ({len(split.test)} matchs)",
            ],
        }
    )
    psi = summary.drift.get("psi") or {}
    drift = (
        h.table(pd.DataFrame({"feature": list(psi), "PSI": list(psi.values())}).head(10))
        if psi
        else h.paragraph("Première campagne de l'expérience : pas encore de référence pour mesurer la dérive.")
    )
    durations = pd.DataFrame(
        {
            "étape": [k.removeprefix("duration/") for k in summary.durations],
            "secondes": list(summary.durations.values()),
        }
    )
    prefect = ctx.get("prefect.flow_run_url", "")
    reproducibility = pd.DataFrame(
        {
            "élément": ["commit Git", "modifications non commitées", "uv.lock (SHA-256)", "Python", "run Prefect"],
            "valeur": [
                ctx.get("git.commit", "?"),
                ctx.get("git.dirty", "?"),
                ctx.get("uv_lock.sha256", "?"),
                ctx.get("python.version", "?"),
                prefect or "—",
            ],
        }
    )
    body = [
        h.cards(key_figures),
        h.section(
            "Candidats",
            h.paragraph("Classés sur la validation ; la ligne surlignée est le candidat retenu."),
            f'<img class="wide" alt="candidats" src="{h.figure_src(figures["candidates_validation"])}">',
            h.table(table, highlight="selected"),
        ),
        h.section("Baselines", h.table(baselines)),
        h.section(
            "Le retenu dans le temps", f'<img class="wide" alt="mois" src="{h.figure_src(figures["monthly_best"])}">'
        ),
        _decision_section(summary),
        h.section("Données", h.table(data)),
        h.section(
            "Dérive par rapport à la campagne précédente",
            h.paragraph(f"PSI maximal : {summary.drift.get('max_psi', 0):.3f} (alerte au-delà de 0,2)." if psi else ""),
            drift,
        ),
        h.section(
            "Durées",
            h.table(durations, digits=1),
            h.paragraph(f"Mémoire maximale du processus : {summary.peak_rss_mb:.0f} Mo."),
        ),
        h.section("Reproductibilité", h.table(reproducibility)),
    ]
    trained = "entraînés" if len(summary.results) > 1 else "entraîné"
    lead = f"{_count(len(summary.results), 'candidat')} {trained} sur les matchs validés jusqu'au {split.end}."
    return h.page("Campagne d'entraînement", lead, body)


PART_ORDER = ("train", "validation", "test")


def _rounded(metrics: Mapping[str, float]) -> dict[str, float | int]:
    values: dict[str, float | int] = {k: metrics[k] for k in ("log_loss", "brier", "accuracy", "roc_auc", "ece")}
    values["matchs"] = int(metrics["rows"])
    return values


# --------------------------------------------------------------------------- journalisation


@dataclass(frozen=True)
class CandidateArtifacts:
    """Ce que le rapport d'un candidat réutilise ensuite pour la campagne."""

    model_info: ModelInfo
    monthly: pd.DataFrame


@dataclass(frozen=True)
class CandidateResult:
    """Ce qu'il reste d'un candidat une fois journalisé."""

    name: str
    run_id: str
    model_uri: str
    validation: dict[str, float]
    test: dict[str, float]
    fit_seconds: float
    monthly: pd.DataFrame
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)  # test, toutes dimensions (règle de mise en service)


@dataclass(frozen=True)
class CampaignSummary:
    """Tout ce que le bilan de campagne journalise (métriques, tableau, graphiques, rapport HTML)."""

    results: Sequence[CandidateResult]
    best: CandidateResult
    baselines: Mapping[str, Evaluation]
    production: Evaluation | None
    comparison: str
    registered: RegisteredVersion
    durations: Mapping[str, float]
    workers: int
    peak_rss_mb: float
    dataset: Dataset
    split: Split
    drift: Mapping[str, Any]
    context: Mapping[str, str]
    failed: Sequence[str] = ()
    decision: GateDecision | None = None
    event: RegistryEvent | None = None

    @property
    def fit_sum(self) -> float:
        """Somme des durées d'entraînement individuelles."""
        return sum(r.fit_seconds for r in self.results)

    @property
    def best_monthly(self) -> pd.DataFrame:
        """Performance mensuelle du candidat retenu."""
        return self.best.monthly

    def candidates_table(self, metric: str) -> pd.DataFrame:
        """Un candidat par ligne, du meilleur au moins bon sur la validation."""
        return (
            pd.DataFrame(
                [
                    {
                        "candidate": r.name,
                        f"validation_{metric}": r.validation[metric],
                        f"test_{metric}": r.test[metric],
                        "validation_roc_auc": r.validation["roc_auc"],
                        "validation_ece": r.validation["ece"],
                        "fit_seconds": r.fit_seconds,
                        "run_id": r.run_id,
                        "selected": r.run_id == self.best.run_id,
                    }
                    for r in self.results
                ]
            )
            .sort_values(f"validation_{metric}")
            .reset_index(drop=True)
        )

    def metrics(self, metric: str) -> dict[str, float]:
        """Métriques du run parent, rangées par section."""
        values: dict[str, float] = dict(self.durations)
        for name, evaluation in self.baselines.items():
            values.update({k: v for k, v in evaluation.flat_metrics(f"baseline_{name}").items() if "/train_" not in k})
        values.update({f"best/validation_{k}": v for k, v in self.best.validation.items()})
        values.update({f"best/test_{k}": v for k, v in self.best.test.items()})
        if self.production is not None:
            values.update({f"production/test_{k}": v for k, v in self.production.metrics["test"].items()})
            values[f"production/best_minus_production_test_{metric}"] = (
                self.best.test[metric] - self.production.metrics["test"][metric]
            )
        if self.decision is not None:
            values["gate/passed"] = float(self.decision.passed)
            values["gate/rules_failed"] = float(len(self.decision.failed_rules))
            if self.decision.delta_log_loss is not None:
                values["gate/delta_log_loss"] = self.decision.delta_log_loss
        wall = self.durations["duration/training_wall_s"]
        values.update(
            {
                "parallel/workers": float(self.workers),
                "parallel/training_wall_s": wall,
                "parallel/fit_sum_s": self.fit_sum,
                "parallel/fit_sum_over_wall": self.fit_sum / wall if wall else 0.0,
                "cost/peak_rss_mb": self.peak_rss_mb,
            }
        )
        return values


class Reporter:
    """Écrit dans MLflow tout ce qui permet de comprendre, comparer et réentraîner un candidat.

    Les métriques sont rangées par **section** (`test/`, `validation/`, `cost/`, `segments/`…) : MLflow en fait des
    groupes repliables dans l'onglet « Model metrics ». Les séries (`monthly/`, `learning/`) y deviennent des courbes.
    """

    def __init__(self, tracker: MlflowTracker, config: TrainConfig) -> None:
        self.tracker = tracker
        self.config = config

    def log_candidate(
        self,
        run_id: str,
        fitted: FittedCandidate,
        evaluation: Evaluation,
        baselines: Mapping[str, Evaluation],
        segments: Sequence[pd.DataFrame],
        diagnostics: Mapping[str, Any],
        split: Split,
        dataset: Dataset,
    ) -> CandidateArtifacts:
        """Paramètres, métriques, séries, graphiques, rapport HTML puis le modèle ; le run est fermé à la fin."""
        t, cfg = self.tracker, self.config
        spec = fitted.spec
        t.log_params(
            run_id,
            {
                "model.kind": spec.kind.value,
                **{f"model.{k}": v for k, v in spec.params.items()},
                "seed": spec.seed,
                "features.version": FEATURES_VERSION,
                "features.columns": ",".join(FEATURE_COLUMNS),
                **split.describe(),
            },
        )
        segment_table = pd.concat(segments, ignore_index=True)
        monthly: pd.DataFrame = diagnostics["monthly"]
        curve: pd.DataFrame | None = diagnostics.get("learning")
        t.log_metrics(run_id, self._candidate_metrics(fitted, evaluation, baselines, segment_table, diagnostics))
        _log_monthly(t, run_id, monthly, "monthly")
        if curve is not None:
            t.log_series(
                run_id, "learning/train_log_loss", list(zip(curve["iteration"], curve["train_log_loss"], strict=True))
            )
            t.log_series(
                run_id,
                "learning/validation_log_loss",
                list(zip(curve["iteration"], curve["validation_log_loss"], strict=True)),
            )

        figures = self._candidate_figures(evaluation, segment_table, diagnostics, split)
        for name, figure in figures.items():
            t.log_figure(run_id, figure, f"plots/{name}.png")
        importance: pd.DataFrame = diagnostics["importance"]
        t.log_frame(run_id, segment_table, "evaluation/segments_test.csv")
        t.log_frame(run_id, importance, "evaluation/feature_importance.csv")
        t.log_frame(run_id, diagnostics["worst"], "evaluation/worst_predictions_test.csv")
        t.log_frame(run_id, monthly, "evaluation/monthly.csv")
        y_test, p_test = Split.labels(split.test).to_numpy(), evaluation.probabilities["test"]
        t.log_frame(
            run_id,
            calibration_table(y_test, p_test, cfg.evaluation.calibration_bins),
            "evaluation/calibration_test.csv",
        )
        if curve is not None:
            t.log_frame(run_id, curve, "evaluation/learning_curve.csv")
        card = model_card(fitted, evaluation, baselines, split, dataset, importance, cfg)
        t.log_text(run_id, card, "model_card.md")
        t.log_text(
            run_id,
            candidate_html(fitted, evaluation, baselines, segment_table, diagnostics, split, figures, card),
            "rapport.html",
        )

        signature, example = model_contract(fitted.pipeline, Split.features(split.validation))
        info = t.log_model(run_id, fitted.pipeline, signature, example, trusted_types(fitted.pipeline))
        return CandidateArtifacts(model_info=info, monthly=monthly)

    @staticmethod
    def _candidate_metrics(
        fitted: FittedCandidate,
        evaluation: Evaluation,
        baselines: Mapping[str, Evaluation],
        segment_table: pd.DataFrame,
        diagnostics: Mapping[str, Any],
    ) -> dict[str, float]:
        metrics = dict(evaluation.flat_metrics())
        metrics["cost/fit_s"] = fitted.fit_seconds
        metrics.update(diagnostics["costs"])
        for name, baseline in baselines.items():
            for part in ("validation", "test"):
                gap = evaluation.metrics[part]["log_loss"] - baseline.metrics[part]["log_loss"]
                metrics[f"vs_baselines/{part}_log_loss_minus_{name}"] = gap
        for record in segment_table.dropna(subset=["log_loss"]).to_dict("records"):
            metrics[metric_key("segments", record["dimension"], record["segment"])] = float(record["log_loss"])
        return metrics

    def _candidate_figures(
        self, evaluation: Evaluation, segment_table: pd.DataFrame, diagnostics: Mapping[str, Any], split: Split
    ) -> dict[str, Figure]:
        y, p = Split.labels(split.test).to_numpy(), evaluation.probabilities["test"]
        test = evaluation.metrics["test"]
        figures = {
            "monthly": monthly_figure(diagnostics["monthly"], "Log loss mois par mois (validation puis test)"),
            "calibration_test": calibration_figure(
                y, p, self.config.evaluation.calibration_bins, f"Calibration (test) — ECE {test['ece']:.3f}"
            ),
            "roc_test": roc_figure(y, p, test["roc_auc"], "ROC (test)"),
            "confusion_test": confusion_figure(y, p, "Matrice de confusion (test, seuil 0,5)"),
            "probabilities_test": distribution_figure(y, p, "Probabilités prédites (test)"),
            "feature_importance": importance_figure(
                diagnostics["importance"], "Importance par permutation (validation)"
            ),
            "segments_test": segments_figure(segment_table, test["log_loss"], "Log loss par segment (test)"),
        }
        if diagnostics.get("learning") is not None:
            figures["learning_curve"] = learning_figure(diagnostics["learning"], "Courbe d'apprentissage")
        return figures

    def log_campaign(self, run_id: str, summary: CampaignSummary) -> None:
        """Bilan du run parent : tableau et graphique des candidats, métriques par section, rapport HTML."""
        t, metric = self.tracker, self.config.selection_metric
        table = summary.candidates_table(metric)
        t.log_frame(run_id, table, "candidates.csv")
        reference = {name: ev.metrics["validation"][metric] for name, ev in summary.baselines.items()}
        figures = {
            "candidates_validation": candidates_figure(table, reference, metric),
            "monthly_best": monthly_figure(summary.best_monthly, f"Retenu ({summary.best.name}) mois par mois"),
        }
        for name, figure in figures.items():
            t.log_figure(run_id, figure, f"plots/{name}.png")
        t.log_frame(run_id, summary.best_monthly, "evaluation/monthly_best.csv")
        t.log_metrics(run_id, summary.metrics(metric))
        _log_monthly(t, run_id, summary.best_monthly, "monthly_best")
        t.set_tags(
            run_id,
            {
                "best.candidate": summary.best.name,
                "best.run_id": summary.best.run_id,
                "best.model_uri": summary.best.model_uri,
                "registry.model": summary.registered.name,
                "registry.version": summary.registered.version,
                "production.comparison": summary.comparison,
                **_decision_tags(summary),
            },
        )
        if summary.decision is not None:
            t.log_dict(run_id, summary.decision.as_dict(), "decision/gate.json")
            t.log_frame(run_id, summary.decision.table(), "decision/rules.csv")
        t.log_text(run_id, campaign_html(summary, table, figures, metric), "rapport_campagne.html")


def _count(n: int, noun: str) -> str:
    """« 1 candidat », « 8 candidats »."""
    return f"{n} {noun}{'s' if n > 1 else ''}"


def _decision_label(summary: CampaignSummary) -> str:
    if summary.event is None:
        return "lifecycle=candidate (pas de décision)"
    if summary.decision is not None and summary.decision.passed:
        return "mise en production"
    return "refusée : production inchangée"


def _decision_tags(summary: CampaignSummary) -> dict[str, str]:
    if summary.decision is None or summary.event is None:
        return {}
    return {
        "gate.status": summary.decision.status.value,
        "gate.reason": summary.decision.reason[:5000],
        "registry.production_version": summary.event.production_after or "",
    }


def _decision_section(summary: CampaignSummary) -> str:
    """Règle de mise en service : verdict, règle par règle, et ce qui a bougé dans le registre."""
    if summary.decision is None or summary.event is None:
        return h.section("Décision de mise en service", h.paragraph(h.esc(summary.comparison)))
    decision, event, version = summary.decision, summary.event, f"v{summary.registered.version}"
    if decision.passed and event.production_before:
        verdict = f"<strong>Mise en production</strong> : {version} remplace v{h.esc(event.production_before)}"
    elif decision.passed:
        verdict = f"<strong>Mise en production</strong> : {version} est la première version servie"
    else:
        kept = f"v{h.esc(event.production_before)}" if event.production_before else "aucune version"
        verdict = (
            f"<strong>Refusée</strong> : la production reste {kept} ; {version} passe en observation (alias "
            "<code>staging</code>)"
        )
    return h.section(
        "Décision de mise en service",
        h.paragraph(verdict + "."),
        h.table(decision.table(), tones={"OK": "good", "ÉCHEC": "bad", "sans objet": "muted"}),
        h.paragraph(
            "Règles et seuils : <code>configs/promotion.yaml</code> ; stratégie de tagging : "
            "<code>docs/TAGGING.md</code>. « sans objet » : la règle n'a encore rien à comparer (pas de production, "
            "première campagne)."
        ),
    )


def _log_monthly(tracker: MlflowTracker, run_id: str, monthly: pd.DataFrame, section: str) -> None:
    """Séries mois par mois : pas = rang du mois, horodatage = début du mois simulé (axe « Time » de MLflow)."""
    stamps = [int(pd.Timestamp(m).timestamp() * 1000) for m in monthly["month_start"]]
    for column, name in (
        ("log_loss", "log_loss"),
        ("accuracy", "accuracy"),
        ("gap_vs_reference", "log_loss_minus_elo"),
        ("matches", "matches"),
    ):
        points = [
            (step, float(value), stamp) for step, (value, stamp) in enumerate(zip(monthly[column], stamps, strict=True))
        ]
        tracker.log_series(run_id, f"{section}/{name}", points)


def finite(value: float) -> float | None:
    """`None` pour NaN / infini (JSON propre)."""
    return value if math.isfinite(value) else None
