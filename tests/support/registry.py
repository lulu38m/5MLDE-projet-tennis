"""Registre MLflow local peuplé de modèles minuscules : de quoi tester promotions, rejets, rollbacks et rechargement."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from tennis_mlops.features import FEATURES_VERSION
from tennis_mlops.registry.gate import GateDecision, RuleResult
from tennis_mlops.registry.journal import DecisionJournal, journal_experiment
from tennis_mlops.registry.model_registry import ModelRegistry
from tennis_mlops.training.report import model_contract, trusted_types
from tennis_mlops.training.tracking import MlflowTracker
from tests.support.mlflow_local import sqlite_uri, working_directory

MODEL = "atp-registre"


def toy_pipeline(win_rate: float) -> Pipeline:
    """Modèle constant : annonce `win_rate` pour chaque match (versions faciles à distinguer)."""
    wins = round(win_rate * 20)
    labels = np.array([1] * wins + [0] * (20 - wins))
    frame = pd.DataFrame({"elo_diff": np.linspace(-100, 100, 20)})
    return Pipeline([("model", DummyClassifier(strategy="prior"))]).fit(frame, labels)


def probe(model: Pipeline) -> float:
    """La probabilité annoncée par un modèle constant."""
    return float(model.predict_proba(pd.DataFrame({"elo_diff": [0.0]}))[0, 1])


@dataclass
class RegistryWorld:
    """Un serveur MLflow local (SQLite), le registre d'un modèle et son journal de décisions."""

    folder: Path
    tracker: MlflowTracker
    registry: ModelRegistry
    journal: DecisionJournal

    @property
    def uri(self) -> str:
        return self.tracker.tracking_uri

    def add_version(self, win_rate: float, tags: dict[str, str] | None = None) -> str:
        """Journalise un modèle constant dans un run et l'enregistre ; renvoie le numéro de version."""
        model = toy_pipeline(win_rate)
        run_id = self.tracker.start_run(f"jouet {win_rate}", {"kind": "candidate"})
        signature, example = model_contract(model, pd.DataFrame({"elo_diff": [-50.0, 0.0, 50.0]}))
        info = self.tracker.log_model(run_id, model, signature, example, trusted_types(model))
        self.tracker.finish(run_id)
        base = {"lifecycle": "candidate", "features_version": FEATURES_VERSION, "candidate": f"jouet {win_rate}",
                "test.log_loss": f"{0.6 + win_rate / 10:.5f}", "campaign_run_id": run_id}  # fmt: skip
        return self.registry.register(info.model_uri, run_id, base | (tags or {}), "jouet").version

    def tags(self, version: str) -> dict[str, str]:
        return dict(self.registry.get(version).tags)

    def journal_runs(self) -> list[dict[str, str]]:
        experiment = self.tracker.client.get_experiment_by_name(journal_experiment(MODEL))
        assert experiment is not None
        runs = self.tracker.client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"])
        return [dict(r.data.tags) for r in runs]


@contextmanager
def registry_world(folder: Path) -> Iterator[RegistryWorld]:
    """Monde de test autonome, dans `folder`."""
    with working_directory(folder):
        tracker = MlflowTracker(sqlite_uri(folder), "registre-tests", pip_requirements=["scikit-learn==1.6.1"])
        journal = DecisionJournal(MlflowTracker(tracker.tracking_uri, journal_experiment(MODEL), check_server=False))
        yield RegistryWorld(folder, tracker, ModelRegistry(tracker.client, MODEL), journal)


def verdict(passed: bool, compared_to: str | None = None, delta: float | None = None) -> GateDecision:
    """Décision minimale (une règle) pour tester l'application d'un verdict sans tout le calcul."""
    rule = RuleResult("bat_elo", "meilleur que l'Elo pur (log loss)", passed, "0.6000 contre 0.6100", "≤ Elo pur")
    return GateDecision(rules=(rule,), compared_to_version=compared_to, delta_log_loss=delta)
