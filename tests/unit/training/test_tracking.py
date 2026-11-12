import math
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from tennis_mlops.training.tracking import MlflowTracker, TrackingError, code_info
from tests.support.mlflow_local import sqlite_uri, working_directory


@pytest.fixture
def tracker(tmp_path: Path) -> Iterator[MlflowTracker]:
    with working_directory(tmp_path):
        yield MlflowTracker(sqlite_uri(tmp_path), "suivi")


def test_unreachable_server_is_an_explicit_error() -> None:
    with pytest.raises(TrackingError, match="serveur MLflow indisponible"):
        MlflowTracker("http://127.0.0.1:9", "x")


def test_runs_params_metrics_and_artifacts(tracker: MlflowTracker) -> None:
    parent = tracker.start_run("campagne", {"kind": "campaign"})
    child = tracker.start_run("candidat", {"kind": "candidate"}, parent_id=parent)
    tracker.log_params(child, {f"p{i}": i for i in range(150)} | {"long": "x" * 7000})
    skipped = tracker.log_metrics(child, {"ok": 0.5, "nan": math.nan, "inf": math.inf})
    tracker.log_frame(child, pd.DataFrame({"a": [1]}), "evaluation/table.csv")
    sha = tracker.log_parquet(child, pd.DataFrame({"a": [1, 2]}), "data/snapshot.parquet")
    tracker.finish(child)
    tracker.finish(parent)

    run = tracker.client.get_run(child)
    assert skipped == ["inf", "nan"]
    assert run.data.metrics == {"ok": 0.5}
    assert run.data.tags["metrics.not_finite"] == "inf, nan"
    assert len(run.data.params) == 151
    assert len(run.data.params["long"]) == 6000
    assert run.data.tags["mlflow.parentRunId"] == parent
    assert run.info.status == "FINISHED"
    assert len(sha) == 64
    assert {a.path for a in tracker.client.list_artifacts(child, "evaluation")} == {"evaluation/table.csv"}
    assert tracker.run_url(child).endswith(f"/runs/{child}")


def test_series_become_curves(tracker: MlflowTracker) -> None:
    run = tracker.start_run("candidat", {})
    tracker.log_series(
        run, "monthly/log_loss", [(0, 0.61, 1_400_000_000_000), (1, 0.59, 1_402_000_000_000), (2, math.nan)]
    )
    history = tracker.client.get_metric_history(run, "monthly/log_loss")
    assert [(m.step, m.value, m.timestamp) for m in history] == [
        (0, 0.61, 1_400_000_000_000),
        (1, 0.59, 1_402_000_000_000),
    ]


def test_failures_close_every_open_run(tracker: MlflowTracker) -> None:
    parent = tracker.start_run("campagne", {"kind": "campaign"})
    child = tracker.start_run("candidat", {}, parent_id=parent)
    other = tracker.start_run("autre", {}, parent_id=parent)
    tracker.fail_run(other, "candidat", ValueError("C négatif"))
    tracker.fail_run(other, "candidat", ValueError("déjà fermé : ignoré"))
    closed = tracker.fail_open_runs("Trainer.fit", RuntimeError("panne"))

    assert sorted(closed) == sorted([parent, child])
    for run_id in (parent, child, other):
        assert tracker.client.get_run(run_id).info.status == "FAILED"
    tags = tracker.client.get_run(child).data.tags
    assert (tags["failure.step"], tags["failure.cause"]) == ("Trainer.fit", "RuntimeError: panne")
    assert tracker.client.get_run(other).data.tags["failure.cause"] == "ValueError: C négatif"


def test_previous_campaign_and_json_artifacts(tracker: MlflowTracker) -> None:
    first = tracker.start_run("campagne 1", {"kind": "campaign"})
    tracker.log_dict(first, {"rows": 3}, "data/profile_train.json")
    tracker.finish(first)
    second = tracker.start_run("campagne 2", {"kind": "campaign"})
    previous = tracker.previous_campaign(exclude_run_id=second)
    assert previous is not None and previous.info.run_id == first
    assert tracker.read_json_artifact(first, "data/profile_train.json") == {"rows": 3}
    assert tracker.read_json_artifact(first, "absent.json") is None


def test_code_info(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("verrou", encoding="utf-8")
    info = code_info(tmp_path)
    assert info["git.commit"].startswith("inconnu")
    assert len(info["uv_lock.sha256"]) == 64
    assert code_info(tmp_path / "absent")["uv_lock.sha256"] == "absent"


def test_code_info_in_the_docker_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENNIS_GIT_COMMIT", "d266f3d0c0ffee")  # pas de .git dans l'image : révision du build
    info = code_info(tmp_path)
    assert (info["git.commit"], info["mlflow.source.git.commit"]) == ("d266f3d0c0ffee", "d266f3d0c0ffee")
    assert info["git.dirty"] == "inconnu (image Docker)"
