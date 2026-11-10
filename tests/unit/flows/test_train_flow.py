"""Campagnes complètes : données synthétiques en mémoire, MLflow local (SQLite), serveur Prefect de test."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from prefect import State

from tennis_mlops.flows import train
from tennis_mlops.flows.train import CampaignReport, TrainingDependencies
from tennis_mlops.registry.gate import PromotionPolicy
from tennis_mlops.registry.model_registry import ModelRegistry
from tennis_mlops.training.config import CandidateGrid, ModelKind, SplitConfig, TrainConfig
from tennis_mlops.training.tracking import MlflowTracker
from tests.support.mlflow_local import sqlite_uri, working_directory
from tests.support.synthetic import FrameSource, synthetic_matches
from tests.support.training import small_config


@dataclass
class TrainWorld:
    """Une base MLflow locale et des données synthétiques, branchées sur le flow."""

    folder: Path
    tracker: MlflowTracker
    source: FrameSource

    def config(self, **overrides: object) -> str:
        path = self.folder / f"train-{len(list(self.folder.glob('train-*.yaml')))}.yaml"
        path.write_text(yaml.safe_dump(small_config(**overrides).model_dump(mode="json")), encoding="utf-8")
        return str(path)

    def promotion(self, **overrides: object) -> str:
        """Règle de mise en service adaptée aux petites données synthétiques."""
        policy = PromotionPolicy.model_validate({"min_test_rows": 50, "min_segment_rows": 30, **overrides})
        path = self.folder / f"promotion-{len(list(self.folder.glob('promotion-*.yaml')))}.yaml"
        path.write_text(yaml.safe_dump(policy.model_dump()), encoding="utf-8")
        return str(path)

    @property
    def registry(self) -> ModelRegistry:
        return ModelRegistry(self.tracker.client, "atp-tests")

    def decisions(self) -> list[dict[str, str]]:
        experiment = self.tracker.client.get_experiment_by_name("atp-tests-decisions")
        assert experiment is not None
        runs = self.tracker.client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"])
        return [dict(r.data.tags) for r in runs]

    def runs(self, kind: str) -> list[Any]:
        return list(self.tracker.client.search_runs([self.tracker.experiment_id], f"tags.kind = '{kind}'",
                                                    order_by=["attributes.start_time ASC"]))  # fmt: skip


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TrainWorld]:
    with working_directory(tmp_path):
        tracker = MlflowTracker(sqlite_uri(tmp_path), "tests", pip_requirements=["scikit-learn==1.6.1"])
        source = FrameSource(synthetic_matches(surface_effect=1.0))  # des spécialistes : l'Elo pur est battable
        monkeypatch.setattr(train, "dependency_factory", lambda *_: TrainingDependencies(source, tracker))
        yield TrainWorld(tmp_path, tracker, source)


def campaign(
    world: TrainWorld, promotion: dict[str, object] | None = None, **overrides: object
) -> tuple[State[Any], CampaignReport | None]:
    state = train.training_flow(world.config(**overrides), None, 2, world.promotion(**(promotion or {})),
                                return_state=True)  # fmt: skip
    result = state.result(raise_on_failure=False)
    return state, result if isinstance(result, CampaignReport) else None


def test_campaign_trains_compares_and_registers(world: TrainWorld) -> None:
    state, report = campaign(world)
    assert state.is_completed() and report is not None
    assert (report.candidates, report.workers, report.failed_candidates) == (3, 2, [])
    assert report.best_validation <= min(report.baselines_validation.values()) + 0.05
    assert report.registered_version == "1"
    assert report.production_comparison == "aucune version en production"
    assert (report.gate_status, report.production_version) == ("passed", "1")
    assert "première mise en service" in report.gate_reason and "en production" in report.decision

    (parent,) = world.runs("campaign")
    children = world.runs("candidate")
    assert parent.info.status == "FINISHED"
    assert len(children) == 3 and all(c.info.status == "FINISHED" for c in children)
    assert all(c.data.tags["mlflow.parentRunId"] == parent.info.run_id for c in children)
    assert {"parallel/training_wall_s", "baseline_elo/validation_log_loss", "best/test_roc_auc",
            "cost/peak_rss_mb", "duration/training_wall_s", "duration/decision_s", "gate/passed"} <= set(
        parent.data.metrics)  # fmt: skip
    assert (parent.data.tags["gate.status"], parent.data.tags["registry.production_version"]) == ("passed", "1")
    assert parent.data.params["promotion.min_test_rows"] == "50"
    assert parent.data.params["split.train_rows"] == str(report_rows(parent))
    artifacts = {a.path for a in world.tracker.client.list_artifacts(parent.info.run_id, "data")}
    assert {"data/features.parquet", "data/profile_train.json", "data/provenance.json"} <= artifacts
    best = next(c for c in children if c.info.run_id == report.best_run_id)
    assert {"segments/surface.Clay", "cost/latency_1_row_p95_ms", "vs_baselines/validation_log_loss_minus_elo"} <= set(
        best.data.metrics
    )
    client = world.tracker.client
    assert {"rapport_campagne.html", "candidates.csv"} <= {a.path for a in client.list_artifacts(parent.info.run_id)}
    assert {"decision/gate.json", "decision/rules.csv"} <= {
        a.path for a in client.list_artifacts(parent.info.run_id, "decision")
    }
    campaign_html = client.download_artifacts(parent.info.run_id, "rapport_campagne.html", str(world.folder))
    assert "Décision de mise en service" in Path(campaign_html).read_text(encoding="utf-8")
    assert "rapport.html" in {a.path for a in client.list_artifacts(best.info.run_id)}
    assert len(client.get_metric_history(best.info.run_id, "monthly/log_loss")) > 6
    assert len(client.get_metric_history(parent.info.run_id, "monthly_best/log_loss")) > 6
    hgb = next(c for c in children if c.info.run_name.startswith("hgb"))
    assert len(client.get_metric_history(hgb.info.run_id, "learning/validation_log_loss")) == 30
    html_path = client.download_artifacts(best.info.run_id, "rapport.html", str(world.folder))
    assert "Dans le temps" in Path(html_path).read_text(encoding="utf-8")
    version = world.tracker.client.get_model_version("atp-tests", "1")
    tags = version.tags
    assert (version.run_id, tags["lifecycle"], tags["gate_status"], tags["decided_by"]) == (
        report.best_run_id, "production", "passed", "pipeline",
    )  # fmt: skip
    assert tags["campaign_run_id"] == parent.info.run_id and tags["data_validation"] == "passed"
    assert int(tags["n_test"]) > 0 and len(tags["config_hash"]) == 16 and tags["candidate"] == report.best_candidate
    assert world.registry.aliases() == {"production": "1"}
    (decision,) = world.decisions()
    assert (decision["event"], decision["version"], decision["campaign_run_id"]) == (
        "promotion", "1", parent.info.run_id,
    )  # fmt: skip


def report_rows(run: Any) -> int:
    return int(run.data.params["split.train_rows"])


def test_second_campaign_measures_drift_and_compares_with_production(world: TrainWorld) -> None:
    _, first = campaign(world)
    assert first is not None and first.production_version == "1"
    world.source.frame = synthetic_matches(
        weeks=180, seed=8, surface_effect=1.0
    )  # une saison de plus, d'autres joueurs
    state, second = campaign(world, promotion={"max_psi": 10.0, "max_segment_regression": 1.0})
    assert state.is_completed() and second is not None
    assert second.registered_version == "2"
    assert second.production_comparison == "v1"
    parent = world.tracker.client.get_run(second.campaign_run_id)
    assert "drift/max_psi" in parent.data.metrics
    assert "production/best_minus_production_test_log_loss" in parent.data.metrics
    assert world.tracker.read_json_artifact(second.campaign_run_id, "data/drift_vs_previous.json") is not None
    gate = world.tracker.read_json_artifact(second.campaign_run_id, "decision/gate.json")
    assert gate is not None and gate["compared_to_version"] == "1"
    aliases = world.registry.aliases()
    if second.gate_status == "passed":  # le verdict dépend des données ; les alias doivent le refléter
        assert aliases == {"production": "2", "previous": "1"}
    else:
        assert aliases == {"production": "1", "staging": "2"}


def test_a_worse_model_is_registered_but_refused(world: TrainWorld) -> None:
    campaign(world)
    weak = (CandidateGrid(model=ModelKind.LOGISTIC_REGRESSION, grid={"C": (0.00001,)}),)
    state, report = campaign(world, candidates=weak)
    assert state.is_completed() and report is not None  # un rejet est une décision, pas un échec
    assert (report.registered_version, report.gate_status, report.production_version) == ("2", "failed", "1")
    assert "Elo pur" in report.gate_reason and "refusée" in report.decision
    assert world.registry.aliases() == {"production": "1", "staging": "2"}
    tags = world.registry.get("2").tags
    assert (tags["lifecycle"], tags["gate_status"], tags["compared_to_version"]) == ("rejected", "failed", "1")
    assert float(tags["delta_log_loss"]) > 0
    assert [d["event"] for d in world.decisions()] == ["promotion", "rejet"]


def test_a_failing_candidate_does_not_stop_the_campaign(world: TrainWorld) -> None:
    candidates = (CandidateGrid(model=ModelKind.LOGISTIC_REGRESSION, grid={"C": (-1.0, 1.0)}),)
    state, report = campaign(world, candidates=candidates)
    assert state.is_completed() and report is not None
    assert report.failed_candidates == ["logreg C=-1.0"]
    failed = next(r for r in world.runs("candidate") if r.info.run_name == "logreg C=-1.0")
    assert failed.info.status == "FAILED"
    assert failed.data.tags["failure.step"] == "candidat"


def test_too_little_data_fails_and_closes_the_campaign_run(world: TrainWorld) -> None:
    split = SplitConfig(train_start=small_config().split.train_start, validation_months=6, test_months=6,
                        min_rows=100_000)  # fmt: skip
    state, _ = campaign(world, split=split)
    assert state.is_failed()
    (parent,) = world.runs("campaign")
    assert parent.info.status == "FAILED"
    assert parent.data.tags["failure.step"] == "TemporalSplitter.split"
    assert "trop petite" in parent.data.tags["failure.cause"]
    assert world.runs("candidate") == []


def test_command_line_exit_codes(world: TrainWorld) -> None:
    assert train.main(["--config", world.config(), "--workers", "1"]) == train.EXIT_OK
    bad = small_config(split=SplitConfig(train_start=small_config().split.train_start, validation_months=6,
                                         test_months=6, min_rows=100_000))  # fmt: skip
    path = world.folder / "bad.yaml"
    path.write_text(yaml.safe_dump(bad.model_dump(mode="json")), encoding="utf-8")
    assert train.main(["--config", str(path)]) == train.EXIT_FAILED


def test_unreachable_mlflow_stops_before_any_computation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def unreachable(config: TrainConfig, _: object) -> TrainingDependencies:
        return TrainingDependencies(FrameSource(synthetic_matches()), MlflowTracker("http://127.0.0.1:9", "x"))

    monkeypatch.setattr(train, "dependency_factory", unreachable)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(small_config().model_dump(mode="json")), encoding="utf-8")
    state = train.training_flow(str(path), None, 1, None, return_state=True)
    assert state.is_failed()
    assert "serveur MLflow indisponible" in str(state.message)
