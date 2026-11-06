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

    def runs(self, kind: str) -> list[Any]:
        return list(self.tracker.client.search_runs([self.tracker.experiment_id], f"tags.kind = '{kind}'",
                                                    order_by=["attributes.start_time ASC"]))  # fmt: skip


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TrainWorld]:
    with working_directory(tmp_path):
        tracker = MlflowTracker(sqlite_uri(tmp_path), "tests", pip_requirements=["scikit-learn==1.6.1"])
        source = FrameSource(synthetic_matches())
        monkeypatch.setattr(train, "dependency_factory", lambda *_: TrainingDependencies(source, tracker))
        yield TrainWorld(tmp_path, tracker, source)


def campaign(world: TrainWorld, **overrides: object) -> tuple[State[Any], CampaignReport | None]:
    state = train.training_flow(world.config(**overrides), None, 2, return_state=True)
    result = state.result(raise_on_failure=False)
    return state, result if isinstance(result, CampaignReport) else None


def test_campaign_trains_compares_and_registers(world: TrainWorld) -> None:
    state, report = campaign(world)
    assert state.is_completed() and report is not None
    assert (report.candidates, report.workers, report.failed_candidates) == (3, 2, [])
    assert report.best_validation <= min(report.baselines_validation.values()) + 0.05
    assert report.registered_version == "1"
    assert report.production_comparison.startswith("aucune version")

    (parent,) = world.runs("campaign")
    children = world.runs("candidate")
    assert parent.info.status == "FINISHED"
    assert len(children) == 3 and all(c.info.status == "FINISHED" for c in children)
    assert all(c.data.tags["mlflow.parentRunId"] == parent.info.run_id for c in children)
    assert {"parallel/training_wall_s", "baseline_elo/validation_log_loss", "best/test_roc_auc",
            "cost/peak_rss_mb", "duration/training_wall_s"} <= set(parent.data.metrics)  # fmt: skip
    assert parent.data.params["split.train_rows"] == str(report_rows(parent))
    artifacts = {a.path for a in world.tracker.client.list_artifacts(parent.info.run_id, "data")}
    assert {"data/features.parquet", "data/profile_train.json", "data/provenance.json"} <= artifacts
    best = next(c for c in children if c.info.run_id == report.best_run_id)
    assert {"segments/surface.Clay", "cost/latency_1_row_p95_ms", "vs_baselines/validation_log_loss_minus_elo"} <= set(
        best.data.metrics
    )
    client = world.tracker.client
    assert {"rapport_campagne.html", "candidates.csv"} <= {a.path for a in client.list_artifacts(parent.info.run_id)}
    assert "rapport.html" in {a.path for a in client.list_artifacts(best.info.run_id)}
    assert len(client.get_metric_history(best.info.run_id, "monthly/log_loss")) > 6
    assert len(client.get_metric_history(parent.info.run_id, "monthly_best/log_loss")) > 6
    hgb = next(c for c in children if c.info.run_name.startswith("hgb"))
    assert len(client.get_metric_history(hgb.info.run_id, "learning/validation_log_loss")) == 30
    html_path = client.download_artifacts(best.info.run_id, "rapport.html", str(world.folder))
    assert "Dans le temps" in Path(html_path).read_text(encoding="utf-8")
    version = world.tracker.client.get_model_version("atp-tests", "1")
    assert (version.run_id, version.tags["lifecycle"], version.tags["qualification"]) == (
        report.best_run_id, "candidate", "pending",
    )  # fmt: skip


def report_rows(run: Any) -> int:
    return int(run.data.params["split.train_rows"])


def test_second_campaign_measures_drift_and_compares_with_production(world: TrainWorld) -> None:
    _, first = campaign(world)
    assert first is not None
    world.tracker.client.set_registered_model_alias("atp-tests", "production", "1")
    world.tracker.client.set_model_version_tag("atp-tests", "1", "features_version", "1")
    world.source.frame = synthetic_matches(weeks=180, seed=8)  # une saison de plus, d'autres joueurs
    state, second = campaign(world)
    assert state.is_completed() and second is not None
    assert second.registered_version == "2"
    assert second.production_comparison == "version 1"
    parent = world.tracker.client.get_run(second.campaign_run_id)
    assert "drift/max_psi" in parent.data.metrics
    assert "production/best_minus_production_test_log_loss" in parent.data.metrics
    assert world.tracker.read_json_artifact(second.campaign_run_id, "data/drift_vs_previous.json") is not None


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
    state = train.training_flow(str(path), None, 1, return_state=True)
    assert state.is_failed()
    assert "serveur MLflow indisponible" in str(state.message)
