"""Entraînement d'un candidat : ajustement du pipeline sur la seule partition d'entraînement, chronométré."""

from __future__ import annotations

import time
from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from tennis_mlops.training.pipeline import CandidateSpec
from tennis_mlops.training.split import Split


@dataclass
class FittedCandidate:
    """Un pipeline ajusté et ce qu'il a coûté."""

    spec: CandidateSpec
    pipeline: Pipeline
    fit_seconds: float

    @property
    def name(self) -> str:
        """Nom du candidat (nom du run MLflow enfant)."""
        return self.spec.name


class Trainer:
    """Ajuste un pipeline ; ni la validation ni le test ne sont vus ici."""

    def fit(self, spec: CandidateSpec, pipeline: Pipeline, split: Split) -> FittedCandidate:
        """`pipeline.fit` sur le train, durée mesurée."""
        started = time.perf_counter()
        pipeline.fit(Split.features(split.train), Split.labels(split.train))
        return FittedCandidate(spec=spec, pipeline=pipeline, fit_seconds=time.perf_counter() - started)
