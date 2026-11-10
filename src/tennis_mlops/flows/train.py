"""Flow Prefect d'entraînement : une campagne = une grille de candidats entraînés **en parallèle**, tracés dans MLflow.

```
DatasetBuilder.build ─► TemporalSplitter.split ─► profil / dérive ─► baselines
   └─► pour chaque candidat, en parallèle (.map) :
         PipelineFactory.create ─► Trainer.fit ─► Evaluator.evaluate ─┬─► segments (5, en parallèle)
                                                                       └─► diagnostics (coûts, importance, erreurs)
                                                                              └─► Reporter.log_candidate (run enfant)
   └─► sélection sur la validation ─► comparaison à la production ─► ModelRegistry.register
         └─► règle de mise en service (configs/promotion.yaml) ─► Promoter.apply : alias `production` déplacé ou
             version refusée (`staging`) ─► journal des décisions ─► bilan (run parent)
```

- **MLflow** : un run parent « campagne » (données, config, baselines, comparaison des candidats) et un run enfant
  par candidat (hyperparamètres, métriques, graphiques, fiche modèle, modèle). Les runs sont ouverts avant le
  calcul : un échec est rattaché au bon run, fermé en `FAILED` avec l'étape et la cause.
- **Parallélisme** : `ThreadPoolTaskRunner(workers)`. Le bilan trace la durée réelle de l'entraînement et la somme
  des durées individuelles ; `make train WORKERS=1` donne la référence séquentielle.
- **Sélection** sur la validation, **bilan** sur le test : le test ne sert jamais à choisir.
- **Mise en service** : le retenu et la production sont jugés sur le même test ; l'alias `production` n'est déplacé
  que si toutes les règles passent (docs/TAGGING.md). Un rejet n'est pas un échec du flow : c'est une décision.

Lancement : `make train` (ou `python -m tennis_mlops.flows.train`).
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from prefect import State, flow, get_run_logger, task, unmapped
from prefect.cache_policies import NO_CACHE
from prefect.futures import PrefectFuture, wait
from prefect.runtime import flow_run
from prefect.task_runners import TaskRunner, ThreadPoolTaskRunner
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from tennis_mlops.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, ProjectConfig, Settings
from tennis_mlops.features import FEATURES_VERSION
from tennis_mlops.ingestion.store import IngestionStore
from tennis_mlops.log import quiet_training_noise
from tennis_mlops.registry.events import RegistryEvent
from tennis_mlops.registry.gate import (
    DEFAULT_PROMOTION_CONFIG_PATH,
    DataEvidence,
    Gate,
    GateDecision,
    GateInputs,
    ModelScores,
    PromotionPolicy,
)
from tennis_mlops.registry.journal import DecisionJournal
from tennis_mlops.registry.model_registry import ModelRegistry, RegisteredVersion
from tennis_mlops.registry.promotion import Promoter
from tennis_mlops.registry.tagging import version_tags
from tennis_mlops.training import drift as data_drift
from tennis_mlops.training.config import DEFAULT_TRAIN_CONFIG_PATH, TrainConfig
from tennis_mlops.training.dataset import Dataset, DatasetBuilder, MatchSource
from tennis_mlops.training.evaluate import SEGMENT_DIMENSIONS, Evaluation, Evaluator, elo_baseline, rank_baseline
from tennis_mlops.training.pipeline import CandidateSpec, PipelineFactory, expand_grid
from tennis_mlops.training.report import CampaignSummary, CandidateResult, Reporter
from tennis_mlops.training.split import Split, TemporalSplitter
from tennis_mlops.training.tracking import MlflowTracker, code_info
from tennis_mlops.training.trainer import FittedCandidate, Trainer
from tennis_mlops.validation.suites import SUITES_VERSION

FLOW_NAME = "entrainement-atp"


class CampaignReport(BaseModel):
    """Bilan d'une campagne, renvoyé par le flow et affiché par la ligne de commande."""

    campaign_run_id: str
    campaign_url: str
    workers: int
    candidates: int
    failed_candidates: list[str] = []
    selection_metric: str
    best_candidate: str
    best_run_id: str
    best_validation: float
    best_test: float
    baselines_validation: dict[str, float]
    registered_version: str | None = None
    production_comparison: str
    gate_status: str = ""
    gate_reason: str = ""
    production_version: str | None = None
    decision: str = ""
    training_wall_seconds: float
    fit_sum_seconds: float
    data_end: str

    def summary(self) -> str:
        """Quelques lignes pour le terminal."""
        baselines = ", ".join(f"{k} {v:.4f}" for k, v in self.baselines_validation.items())
        return (
            f"campagne terminée : {self.candidates} candidat{'s' if self.candidates > 1 else ''} "
            f"({self.workers} en parallèle), données jusqu'au "
            f"{self.data_end}\n"
            f"  retenu   : {self.best_candidate} — {self.selection_metric} validation {self.best_validation:.4f}, "
            f"test {self.best_test:.4f} (baselines validation : {baselines})\n"
            f"  registre : version {self.registered_version} ; comparée à : {self.production_comparison}\n"
            f"  décision : {self.decision}\n"
            f"  entraînement : {self.training_wall_seconds:.1f} s réels pour {self.fit_sum_seconds:.1f} s cumulés\n"
            f"  MLflow   : {self.campaign_url}"
        )


# --------------------------------------------------------------------------- dépendances (injectables)


@dataclass
class TrainingDependencies:
    """Source des matchs et serveur MLflow ; les tests fournissent les leurs."""

    source: MatchSource
    tracker: MlflowTracker


def default_dependencies(config: TrainConfig, settings: Settings) -> TrainingDependencies:
    """PostgreSQL (matchs validés) et serveur MLflow de la stack."""
    tracker = MlflowTracker(str(settings.mlflow_tracking_uri), config.experiment)
    return TrainingDependencies(source=IngestionStore.from_url(settings.sqlalchemy_url()), tracker=tracker)


DependencyFactory = Callable[[TrainConfig, Settings], TrainingDependencies]
dependency_factory: DependencyFactory = default_dependencies


# --------------------------------------------------------------------------- tasks : données


@task(name="DatasetBuilder.build", cache_policy=NO_CACHE)
def build_dataset(source: MatchSource, config: TrainConfig, pause_s: float) -> Dataset:
    """Matchs validés → features calculées avant chaque match."""
    time.sleep(pause_s)
    dataset = DatasetBuilder(source, config.features).build()
    get_run_logger().info("%d matchs joués → %d lignes de features (%s → %s) en %.1f s", dataset.matches_loaded,
                          len(dataset.frame), dataset.first_date, dataset.last_date, dataset.build_seconds)  # fmt: skip
    return dataset


@task(name="TemporalSplitter.split", cache_policy=NO_CACHE)
def split_dataset(dataset: Dataset, config: TrainConfig, pause_s: float) -> Split:
    """Train / validation / test dans l'ordre du temps."""
    time.sleep(pause_s)
    split = TemporalSplitter(config.split).split(dataset.frame)
    get_run_logger().info("découpage : %s", split.describe())
    return split


@task(name="profil et dérive des données", cache_policy=NO_CACHE)
def log_data(tracker: MlflowTracker, campaign_id: str, dataset: Dataset, split: Split) -> dict[str, Any]:
    """Jeux associés au run (digests), instantané Parquet + SHA-256, profil, dérive vs la campagne précédente."""
    digests = {part: tracker.log_dataset(campaign_id, split.part(part), f"atp-{part}", part) for part in
               ("train", "validation", "test")}  # fmt: skip
    snapshot_sha = tracker.log_parquet(campaign_id, dataset.frame, "data/features.parquet")
    current = data_drift.profile(split.train)
    tracker.log_dict(campaign_id, current, "data/profile_train.json")
    tracker.log_dict(campaign_id, {"lots": dataset.lots, "dataset_versions": list(dataset.dataset_versions)},
                     "data/provenance.json")  # fmt: skip
    previous_run = tracker.previous_campaign(exclude_run_id=campaign_id)
    previous = tracker.read_json_artifact(previous_run.info.run_id, "data/profile_train.json") if previous_run else None
    report: dict[str, Any] = {"previous_campaign": previous_run.info.run_id if previous_run else None}
    if previous:
        report.update(data_drift.drift(previous, split.train))
        tracker.log_dict(campaign_id, report, "data/drift_vs_previous.json")
        over_alert = float(len(report["features_over_alert"]))
        tracker.log_metrics(campaign_id, {"drift/max_psi": report["max_psi"], "drift/features_over_alert": over_alert})
        if report["features_over_alert"]:
            get_run_logger().warning("dérive (PSI > 0,2) : %s", ", ".join(report["features_over_alert"]))
    tracker.log_params(campaign_id, {f"data.digest.{k}": v for k, v in digests.items()})
    tracker.set_tags(campaign_id, {"data.snapshot.sha256": snapshot_sha})
    return report


@task(name="baselines", cache_policy=NO_CACHE)
def evaluate_baselines(evaluator: Evaluator, split: Split, pause_s: float) -> dict[str, Evaluation]:
    """Elo pur (sans apprentissage) et classement seul : ce qu'un modèle doit battre."""
    time.sleep(pause_s)
    return {
        "elo": evaluator.evaluate_probabilities("baseline elo", split, elo_baseline),
        "rang": evaluator.evaluate_probabilities("baseline rang", split, rank_baseline(split.train)),
    }


# --------------------------------------------------------------------------- tasks : un candidat


@task(name="PipelineFactory.create", task_run_name="créer {spec.name}", cache_policy=NO_CACHE)
def create_pipeline(spec: CandidateSpec) -> Pipeline:
    """Prétraitement + modèle du candidat."""
    return PipelineFactory().create(spec)


@task(name="Trainer.fit", task_run_name="entraîner {spec.name}", cache_policy=NO_CACHE)
def fit_candidate(spec: CandidateSpec, pipeline: Pipeline, split: Split, pause_s: float) -> FittedCandidate:
    """Ajustement sur le train seul."""
    time.sleep(pause_s)
    fitted = Trainer().fit(spec, pipeline, split)
    get_run_logger().info("%s entraîné en %.2f s", spec.name, fitted.fit_seconds)
    return fitted


@task(name="Evaluator.evaluate", task_run_name="évaluer {fitted.name}", cache_policy=NO_CACHE)
def evaluate_candidate(evaluator: Evaluator, fitted: FittedCandidate, split: Split, pause_s: float) -> Evaluation:
    """Métriques sur train, validation et test."""
    time.sleep(pause_s)
    return evaluator.evaluate(fitted.name, fitted.pipeline, split)


@task(name="Evaluator.segments", task_run_name="segment {dimension} · {evaluation.name}", cache_policy=NO_CACHE)
def evaluate_segment(evaluator: Evaluator, evaluation: Evaluation, split: Split, dimension: str) -> pd.DataFrame:
    """Métriques du test pour une dimension (surface, niveau, tour, écart de classement, Grand Chelem)."""
    return evaluator.segments(evaluation, split, dimension)


@task(name="Evaluator.diagnostics", task_run_name="diagnostics {fitted.name}", cache_policy=NO_CACHE)
def diagnose(
    evaluator: Evaluator,
    fitted: FittedCandidate,
    evaluation: Evaluation,
    baselines: Mapping[str, Evaluation],
    split: Split,
    seed: int,
) -> dict[str, Any]:
    """Coûts, importance par permutation, pires prédictions, mois par mois, courbe d'apprentissage."""
    return {
        "costs": evaluator.costs(fitted.pipeline, split),
        "importance": evaluator.feature_importance(fitted.pipeline, split, seed),
        "worst": evaluator.worst_predictions(evaluation, split),
        "monthly": evaluator.monthly(evaluation, baselines["elo"], split),
        "learning": evaluator.learning_curve(fitted.pipeline, split),
    }


@task(name="Reporter.log", task_run_name="MLflow {fitted.name}", cache_policy=NO_CACHE)
def report_candidate(
    reporter: Reporter,
    run_id: str,
    fitted: FittedCandidate,
    evaluation: Evaluation,
    baselines: Mapping[str, Evaluation],
    segments: list[pd.DataFrame],
    diagnostics: Mapping[str, Any],
    split: Split,
    dataset: Dataset,
) -> CandidateResult:
    """Tout le run enfant (rapport HTML et modèle compris) ; le run est fermé avec succès."""
    logged = reporter.log_candidate(run_id, fitted, evaluation, baselines, segments, diagnostics, split, dataset)
    return CandidateResult(
        name=fitted.name,
        run_id=run_id,
        model_uri=logged.model_info.model_uri,
        validation=evaluation.metrics["validation"],
        test=evaluation.metrics["test"],
        fit_seconds=fitted.fit_seconds,
        monthly=logged.monthly,
        segments=pd.concat(segments, ignore_index=True),
    )


# --------------------------------------------------------------------------- tasks : décision


@dataclass(frozen=True)
class ProductionComparison:
    """Le modèle en service, jugé sur le même test que le candidat (s'il existe et parle les mêmes features)."""

    note: str
    version: str | None = None
    evaluation: Evaluation | None = None
    segments: pd.DataFrame | None = None


@task(name="comparaison avec la production", cache_policy=NO_CACHE)
def compare_with_production(
    registry: ModelRegistry, evaluator: Evaluator, split: Split, dimension: str
) -> ProductionComparison:
    """Évalue le modèle en service sur le même test, s'il existe et parle les mêmes features."""
    production = registry.production()
    if production is None:
        return ProductionComparison("aucune version en production")
    version = str(production.version.version)
    version_features = production.version.tags.get("features_version")
    if version_features != FEATURES_VERSION:
        return ProductionComparison(f"v{version} incomparable (features v{version_features})", version)
    evaluation = evaluator.evaluate(f"production v{version}", production.model, split)
    return ProductionComparison(f"v{version}", version, evaluation, evaluator.segments(evaluation, split, dimension))


@task(name="ModelRegistry.register", cache_policy=NO_CACHE)
def register_best(
    registry: ModelRegistry, config: TrainConfig, best: CandidateResult, tags: Mapping[str, str]
) -> RegisteredVersion:
    """Nouvelle version du modèle retenu, `lifecycle=candidate` jusqu'à la décision."""
    description = (
        f"{best.name} — {config.selection_metric} validation {best.validation[config.selection_metric]:.4f}, "
        f"test {best.test[config.selection_metric]:.4f}. Campagne {tags['campaign_run_id']}."
    )
    return registry.register(best.model_uri, best.run_id, tags, description)


@task(name="règle de mise en service", cache_policy=NO_CACHE)
def decide_promotion(gate: Gate, inputs: GateInputs) -> GateDecision:
    """Verdict règle par règle (configs/promotion.yaml), sur le même test pour le candidat et la production."""
    decision = gate.decide(inputs)
    logger = get_run_logger()
    for rule in decision.rules:
        logger.info("  %-50s %-10s %s", rule.label, "OK" if rule.passed else "ÉCHEC", rule.observed)
    return decision


@task(name="Promoter.apply", cache_policy=NO_CACHE)
def apply_decision(
    promoter: Promoter, journal: DecisionJournal, version: str, decision: GateDecision, campaign_id: str
) -> RegistryEvent:
    """Déplace les alias (ou refuse la version), puis inscrit la décision dans le journal MLflow."""
    event = promoter.apply(version, decision, campaign_id)
    logger = get_run_logger()
    logger.info(event.summary())
    if problem := journal.try_record(event):
        logger.warning(problem)
    return event


# --------------------------------------------------------------------------- flow


@flow(name=FLOW_NAME)
def training_flow(
    train_config_path: str | None = None,
    project_config_path: str | None = None,
    workers: int | None = None,
    promotion_config_path: str | None = None,
) -> CampaignReport:
    """Entraîne la grille de candidats sur les matchs validés, choisit, compare, enregistre, décide."""
    quiet_training_noise()
    config = TrainConfig.from_yaml(Path(train_config_path) if train_config_path else DEFAULT_TRAIN_CONFIG_PATH)
    project = ProjectConfig.from_yaml(Path(project_config_path) if project_config_path else DEFAULT_CONFIG_PATH)
    policy = PromotionPolicy.from_yaml(
        Path(promotion_config_path) if promotion_config_path else DEFAULT_PROMOTION_CONFIG_PATH
    )
    settings = Settings()
    workers = workers or config.workers
    deps = dependency_factory(config, settings)
    tracker, logger = deps.tracker, get_run_logger()
    context = _campaign_tags(settings)
    campaign_id = tracker.start_run(f"campagne {time.strftime('%Y-%m-%d %H:%M')}", context)
    logger.info("campagne MLflow : %s", tracker.run_url(campaign_id))
    progress = Progress()
    try:
        _log_configuration(tracker, campaign_id, config, project, workers)
        _log_policy(tracker, campaign_id, policy)
        campaign = Campaign(campaign_id, context, policy)
        report = _campaign(deps, config, project.demo.sleep_s, workers, campaign, progress)
        progress.step = "bilan"
        tracker.finish(campaign_id)
    except BaseException as exc:
        failed = tracker.fail_open_runs(progress.step, exc)
        logger.error("campagne en échec (%s) : %s — runs fermés en FAILED : %s", progress.step, exc, ", ".join(failed))
        raise
    logger.info(report.summary())
    return report


@dataclass
class Progress:
    """Étape en cours, pour fermer les runs en `FAILED` avec la bonne cause."""

    step: str = "configuration"


@dataclass(frozen=True)
class Campaign:
    """Le run parent et son contexte (révision Git, run Prefect…)."""

    run_id: str
    context: Mapping[str, str]
    policy: PromotionPolicy


def _campaign(
    deps: TrainingDependencies, config: TrainConfig, pause: float, workers: int, campaign: Campaign, progress: Progress
) -> CampaignReport:
    tracker, logger = deps.tracker, get_run_logger()
    campaign_id = campaign.run_id
    evaluator, reporter = Evaluator(config.evaluation), Reporter(tracker, config)
    registry = ModelRegistry(tracker.client, config.registered_model)
    durations: dict[str, float] = {}

    def timed(name: str, started: float) -> None:
        durations[f"duration/{name}_s"] = time.perf_counter() - started

    progress.step = "DatasetBuilder.build"
    started = time.perf_counter()
    dataset = build_dataset(deps.source, config, pause)
    timed("dataset", started)
    progress.step = "TemporalSplitter.split"
    started = time.perf_counter()
    split = split_dataset(dataset, config, pause)
    timed("split", started)
    tracker.log_params(campaign_id, {**split.describe(), "data.matches_loaded": dataset.matches_loaded,
                                     "data.rows": len(dataset.frame), "data.first_date": dataset.first_date,
                                     "data.last_date": dataset.last_date,
                                     "data.dataset_versions": ",".join(dataset.dataset_versions),
                                     "data.players": _players(dataset.frame),
                                     "data.tournaments": int(dataset.frame["tourney_id"].nunique())})  # fmt: skip
    progress.step = "profil, dérive et baselines"
    started = time.perf_counter()
    drift = log_data(tracker, campaign_id, dataset, split)
    baselines = evaluate_baselines(evaluator, split, pause)
    timed("data_report_and_baselines", started)

    specs = expand_grid(config)
    child_tags = {"kind": "candidate", "features_version": FEATURES_VERSION, **code_info(PROJECT_ROOT)}
    run_ids = [tracker.start_run(spec.name, {**child_tags, "model.kind": spec.kind.value}, parent_id=campaign_id)
               for spec in specs]  # fmt: skip
    logger.info("%d candidats, %d en parallèle", len(specs), workers)

    progress.step = "entraînement des candidats"
    started = time.perf_counter()
    pipelines = create_pipeline.map(specs)
    fitted = fit_candidate.map(specs, pipelines, unmapped(split), unmapped(pause))
    wait(list(fitted))
    timed("training_wall", started)
    progress.step = "évaluation et journalisation des candidats"
    started = time.perf_counter()
    evaluations = evaluate_candidate.map(unmapped(evaluator), fitted, unmapped(split), unmapped(pause))
    dims = SEGMENT_DIMENSIONS
    segment_futures = evaluate_segment.map(unmapped(evaluator), [e for e in evaluations for _ in dims],
                                           unmapped(split), [d for _ in evaluations for d in dims])  # fmt: skip
    per_candidate = [list(segment_futures[i * len(dims) : (i + 1) * len(dims)]) for i in range(len(specs))]
    diagnostics = diagnose.map(unmapped(evaluator), fitted, evaluations, unmapped(baselines), unmapped(split),
                               unmapped(config.seed))  # fmt: skip
    reports = report_candidate.map(unmapped(reporter), run_ids, fitted, evaluations, unmapped(baselines),
                                   per_candidate, diagnostics, unmapped(split), unmapped(dataset))  # fmt: skip
    wait(list(reports))
    timed("evaluation_and_logging", started)

    results: list[CandidateResult] = []
    failed: list[str] = []
    for i, (spec, run_id) in enumerate(zip(specs, run_ids, strict=True)):
        chain = [pipelines[i], fitted[i], evaluations[i], diagnostics[i], reports[i]]
        if reports[i].state.is_completed():
            results.append(reports[i].result())
            continue
        error = _first_error([future.state for future in chain])
        tracker.fail_run(run_id, "candidat", error)
        failed.append(spec.name)
        logger.error("candidat %s en échec : %s", spec.name, error)
    if not results:
        msg = f"aucun candidat n'a abouti ({', '.join(failed)})"
        raise RuntimeError(msg)

    metric = config.selection_metric
    best = min(results, key=lambda r: r.validation[metric])
    progress.step = "sélection, comparaison et registre"
    started = time.perf_counter()
    dimension = campaign.policy.segment_dimension
    production = compare_with_production(registry, evaluator, split, dimension)
    tags = version_tags(config=config, dataset=dataset, split=split, context=campaign.context,
                        campaign_run_id=campaign_id, candidate=best.name, validation_score=best.validation[metric],
                        test_score=best.test[metric])  # fmt: skip
    registered = register_best(registry, config, best, tags)
    timed("selection_and_registry", started)

    progress.step = "règle de mise en service"
    started = time.perf_counter()
    inputs = _gate_inputs(best, baselines, production, dataset, split, drift, dimension)
    decision = decide_promotion(Gate(campaign.policy), inputs)
    journal = DecisionJournal.for_model(tracker.tracking_uri, config.registered_model)
    event = apply_decision(Promoter(registry), journal, registered.version, decision, campaign_id)
    timed("decision", started)

    progress.step = "bilan"
    summary = CampaignSummary(
        results=results, best=best, baselines=baselines, production=production.evaluation, comparison=production.note,
        registered=registered, durations=durations, workers=workers, peak_rss_mb=peak_rss_mb(), dataset=dataset,
        split=split, drift=drift, context=campaign.context, failed=failed, decision=decision, event=event,
    )  # fmt: skip
    reporter.log_campaign(campaign_id, summary)
    return CampaignReport(
        campaign_run_id=campaign_id,
        campaign_url=tracker.run_url(campaign_id),
        workers=workers,
        candidates=len(specs),
        failed_candidates=failed,
        selection_metric=metric,
        best_candidate=best.name,
        best_run_id=best.run_id,
        best_validation=best.validation[metric],
        best_test=best.test[metric],
        baselines_validation={k: v.metrics["validation"][metric] for k, v in baselines.items()},
        registered_version=registered.version,
        production_comparison=production.note,
        gate_status=decision.status.value,
        gate_reason=decision.reason,
        production_version=event.production_after,
        decision=event.summary(),
        training_wall_seconds=durations["duration/training_wall_s"],
        fit_sum_seconds=summary.fit_sum,
        data_end=split.end.isoformat(),
    )


def _gate_inputs(
    best: CandidateResult,
    baselines: Mapping[str, Evaluation],
    production: ProductionComparison,
    dataset: Dataset,
    split: Split,
    drift: Mapping[str, Any],
    dimension: str,
) -> GateInputs:
    """Scores du retenu, des baselines et de la production sur le test commun, et état des données."""
    rows = len(split.test)
    lots = dataset.lots
    production_scores = (
        ModelScores.from_metrics(production.evaluation.metrics["test"], rows, production.segments, dimension)
        if production.evaluation is not None
        else None
    )
    return GateInputs(
        candidate=ModelScores.from_metrics(best.test, rows, best.segments, dimension),
        baselines={name: ModelScores.from_metrics(ev.metrics["test"], rows) for name, ev in baselines.items()},
        data=DataEvidence(
            lots_valid=int(lots.get("lots_valid") or 0),
            lots_quarantined=int(lots.get("lots_quarantined") or 0),
            suite_version=SUITES_VERSION,
            max_psi=float(drift["max_psi"]) if "max_psi" in drift else None,
        ),
        production=production_scores,
        production_version=production.version if production_scores is not None else None,
        production_note=production.note,
    )


def _log_policy(tracker: MlflowTracker, run_id: str, policy: PromotionPolicy) -> None:
    tracker.log_params(run_id, {f"promotion.{k}": v for k, v in policy.model_dump().items()})
    tracker.log_text(run_id, yaml.safe_dump(policy.model_dump(), allow_unicode=True, sort_keys=False),
                     "config/promotion_resolved.yaml")  # fmt: skip


def _first_error(states: Sequence[State[Any]]) -> BaseException:
    """La cause d'origine : la première task en échec de la chaîne (les suivantes n'ont pas pu démarrer)."""
    for state in states:
        if state.is_failed() or state.is_crashed():
            outcome = state.result(raise_on_failure=False)
            return outcome if isinstance(outcome, BaseException) else RuntimeError(str(state.message))
    return RuntimeError("chaîne interrompue sans cause identifiée")


def _log_configuration(
    tracker: MlflowTracker, run_id: str, config: TrainConfig, project: ProjectConfig, workers: int
) -> None:
    resolved = config.model_dump(mode="json")
    tracker.log_text(
        run_id, yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), "config/train_resolved.yaml"
    )
    tracker.log_text(run_id, yaml.safe_dump(project.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
                     "config/project_resolved.yaml")  # fmt: skip
    for name in ("pyproject.toml", "uv.lock"):
        if (PROJECT_ROOT / name).exists():
            tracker.log_file(run_id, PROJECT_ROOT / name, "reproductibilite")
    tracker.log_params(run_id, {
        "seed": config.seed, "workers": workers, "selection_metric": config.selection_metric,
        "features.version": FEATURES_VERSION, "candidates": len(expand_grid(config)),
        **{f"features.{k}": v for k, v in config.features.model_dump().items()},
        **{f"split.config.{k}": v for k, v in config.split.model_dump(mode="json").items()},
        "demo.sleep_s": project.demo.sleep_s,
    })  # fmt: skip


def _campaign_tags(settings: Settings) -> dict[str, str]:
    run_id = flow_run.get_id()
    ui = str(settings.prefect_api_url).rstrip("/").removesuffix("/api")
    return {
        "kind": "campaign",
        "features_version": FEATURES_VERSION,
        "prefect.flow_run_id": str(run_id or ""),
        "prefect.flow_run_url": f"{ui}/runs/flow-run/{run_id}" if run_id else "",
        **code_info(PROJECT_ROOT),
    }


def _players(frame: pd.DataFrame) -> int:
    return int(pd.concat([frame["p1_id"], frame["p2_id"]]).nunique())


def peak_rss_mb() -> float:
    """Mémoire résidente maximale du processus (Mo) ; macOS compte en octets, Linux en kilo-octets."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


# --------------------------------------------------------------------------- ligne de commande

EXIT_OK, EXIT_FAILED = 0, 1


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m tennis_mlops.flows.train [--workers N]` : une campagne ; code de sortie 1 en cas d'échec."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.flows.train", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--promotion-config", type=Path, default=DEFAULT_PROMOTION_CONFIG_PATH)
    parser.add_argument(
        "--workers", type=int, default=None, help="candidats en parallèle (défaut : configs/train.yaml)"
    )
    args = parser.parse_args(argv)
    workers = args.workers or TrainConfig.from_yaml(args.config).workers
    # Les futures du pool de threads sont des PrefectFuture : cast pour l'invariance des génériques de Prefect.
    runner = cast("TaskRunner[PrefectFuture[Any]]", ThreadPoolTaskRunner(max_workers=workers))
    state = training_flow.with_options(task_runner=runner)(
        str(args.config), str(args.project_config), workers, str(args.promotion_config), return_state=True
    )
    if state.is_failed():
        return EXIT_FAILED
    result = state.result(raise_on_failure=False)
    if isinstance(result, CampaignReport):
        print(result.summary())
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
