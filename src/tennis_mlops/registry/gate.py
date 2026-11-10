"""Règle de mise en service : un candidat ne prend la place de la production que si **toutes** les règles passent.

Tout est mesuré sur le **même** jeu de test (les mois les plus récents de la campagne), que ni le candidat ni la
production n'ont vu à l'entraînement. Les seuils sont dans `configs/promotion.yaml` ; ce module est pur (aucun accès
à MLflow) : il prend des scores et rend un verdict détaillé, règle par règle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from tennis_mlops.config import PROJECT_ROOT
from tennis_mlops.registry.tagging import GateStatus, clean

DEFAULT_PROMOTION_CONFIG_PATH = PROJECT_ROOT / "configs" / "promotion.yaml"


class PromotionPolicy(BaseModel):
    """Seuils de la règle de mise en service (`configs/promotion.yaml`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_improvement: float = Field(default=0.0, ge=0)
    min_test_rows: int = Field(default=500, ge=1)
    segment_dimension: str = "surface"
    min_segment_rows: int = Field(default=100, ge=1)
    max_segment_regression: float = Field(default=0.03, ge=0)
    max_psi: float = Field(default=0.25, gt=0)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_PROMOTION_CONFIG_PATH) -> Self:
        """Charge et valide le fichier de règles."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


@dataclass(frozen=True)
class ModelScores:
    """Ce que la règle sait d'un modèle sur le test partagé."""

    log_loss: float
    accuracy: float
    rows: int
    segments: Mapping[str, tuple[float, int]] = field(default_factory=dict)  # segment → (log loss, matchs)

    @classmethod
    def from_metrics(
        cls, test: Mapping[str, float], rows: int, segments: pd.DataFrame | None = None, dimension: str = "surface"
    ) -> ModelScores:
        """Depuis les métriques du test et, si fourni, le tableau de segments de l'évaluateur."""
        per_segment: dict[str, tuple[float, int]] = {}
        if segments is not None and not segments.empty and "log_loss" in segments:
            for row in segments[segments["dimension"] == dimension].to_dict("records"):
                value = float(row.get("log_loss", math.nan))
                if math.isfinite(value):
                    per_segment[str(row["segment"])] = (value, int(row["rows"]))
        return cls(log_loss=float(test["log_loss"]), accuracy=float(test["accuracy"]), rows=rows,
                   segments=per_segment)  # fmt: skip


@dataclass(frozen=True)
class DataEvidence:
    """Ce qu'on sait des données de la campagne : lots validés par Great Expectations, dérive."""

    lots_valid: int
    lots_quarantined: int
    suite_version: str
    max_psi: float | None = None  # None : première campagne, rien à comparer


@dataclass(frozen=True)
class GateInputs:
    """Tout ce que la règle compare."""

    candidate: ModelScores
    baselines: Mapping[str, ModelScores]  # "elo" (Elo pur) et "rang" (le mieux classé gagne)
    data: DataEvidence
    production: ModelScores | None = None
    production_version: str | None = None
    production_note: str = "aucune version en production"


@dataclass(frozen=True)
class RuleResult:
    """Une règle : passée ou non, ce qu'on a mesuré et le seuil."""

    rule: str
    label: str
    passed: bool
    observed: str
    threshold: str
    applicable: bool = True  # False : la règle ne s'applique pas encore (pas de production, première campagne)


@dataclass(frozen=True)
class GateDecision:
    """Le verdict, règle par règle."""

    rules: tuple[RuleResult, ...]
    compared_to_version: str | None = None
    delta_log_loss: float | None = None  # candidat - production sur le test (négatif = le candidat fait mieux)

    @property
    def passed(self) -> bool:
        """Vrai si toutes les règles passent."""
        return all(r.passed for r in self.rules)

    @property
    def status(self) -> GateStatus:
        """`passed` ou `failed`."""
        return GateStatus.PASSED if self.passed else GateStatus.FAILED

    @property
    def failed_rules(self) -> list[RuleResult]:
        """Les règles en échec, dans l'ordre."""
        return [r for r in self.rules if not r.passed]

    @property
    def reason(self) -> str:
        """Une phrase pour le tag `gate_reason` et le terminal."""
        if not self.passed:
            return " ; ".join(f"{r.label} ({r.observed}, seuil {r.threshold})" for r in self.failed_rules)
        if self.compared_to_version is None:
            return "toutes les règles passent (aucune production comparable : première mise en service)"
        return f"toutes les règles passent (log loss {self.delta_log_loss:+.4f} contre v{self.compared_to_version})"

    def tags(self) -> dict[str, str]:
        """Tags de décision posés sur la version."""
        delta = "" if self.delta_log_loss is None else f"{self.delta_log_loss:+.5f}"
        return clean({"gate_status": self.status.value, "gate_reason": self.reason,
                      "compared_to_version": self.compared_to_version or "", "delta_log_loss": delta})  # fmt: skip

    def table(self) -> pd.DataFrame:
        """Une ligne par règle (rapport HTML, CSV)."""
        return pd.DataFrame(
            [
                {"règle": r.label, "résultat": _verdict(r), "mesuré": r.observed, "seuil": r.threshold}
                for r in self.rules
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        """Version JSON (artefact `decision/gate.json`)."""
        return {
            "status": self.status.value,
            "reason": self.reason,
            "compared_to_version": self.compared_to_version,
            "delta_log_loss": self.delta_log_loss,
            "rules": [r.__dict__ for r in self.rules],
        }


def _verdict(rule: RuleResult) -> str:
    if not rule.applicable:
        return "sans objet"
    return "OK" if rule.passed else "ÉCHEC"


class Gate:
    """Applique les règles de `configs/promotion.yaml`."""

    def __init__(self, policy: PromotionPolicy) -> None:
        self.policy = policy

    def decide(self, inputs: GateInputs) -> GateDecision:
        """Verdict détaillé ; la production n'est comparée que si elle existe et parle les mêmes features."""
        rules = (
            self._data_validated(inputs.data),
            self._enough_test_rows(inputs.candidate),
            self._beats_elo(inputs),
            self._accuracy_vs_rank(inputs),
            self._better_than_production(inputs),
            self._no_segment_regression(inputs),
            self._no_strong_drift(inputs.data),
        )
        delta = None if inputs.production is None else inputs.candidate.log_loss - inputs.production.log_loss
        compared = inputs.production_version if inputs.production is not None else None
        return GateDecision(rules=rules, compared_to_version=compared, delta_log_loss=delta)

    # ------------------------------------------------------------------ les règles

    @staticmethod
    def _data_validated(data: DataEvidence) -> RuleResult:
        return RuleResult(
            "donnees_validees",
            "données validées par Great Expectations",
            passed=data.lots_valid > 0,
            observed=f"{data.lots_valid} lots acceptés (suite v{data.suite_version}), {data.lots_quarantined} en "
            "quarantaine exclus",
            threshold="uniquement des lots acceptés par la suite bloquante",
        )

    def _enough_test_rows(self, candidate: ModelScores) -> RuleResult:
        return RuleResult(
            "taille_test",
            "jeu de test assez grand",
            passed=candidate.rows >= self.policy.min_test_rows,
            observed=f"{candidate.rows} matchs",
            threshold=f"≥ {self.policy.min_test_rows}",
        )

    @staticmethod
    def _beats_elo(inputs: GateInputs) -> RuleResult:
        elo = inputs.baselines["elo"]
        return RuleResult(
            "bat_elo",
            "meilleur que l'Elo pur (log loss)",
            passed=inputs.candidate.log_loss <= elo.log_loss,
            observed=f"{inputs.candidate.log_loss:.4f} contre {elo.log_loss:.4f}",
            threshold="≤ Elo pur",
        )

    @staticmethod
    def _accuracy_vs_rank(inputs: GateInputs) -> RuleResult:
        rank = inputs.baselines["rang"]
        return RuleResult(
            "accuracy_classement",
            "accuracy au moins celle du « mieux classé gagne »",
            passed=inputs.candidate.accuracy >= rank.accuracy,
            observed=f"{inputs.candidate.accuracy:.4f} contre {rank.accuracy:.4f}",
            threshold="≥ baseline classement",
        )

    def _better_than_production(self, inputs: GateInputs) -> RuleResult:
        label = "au moins aussi bon que la production (log loss)"
        epsilon = self.policy.min_improvement
        threshold = f"≤ production - {epsilon:g}" if epsilon else "≤ production"
        if inputs.production is None:
            return RuleResult("mieux_que_production", label, True, inputs.production_note, threshold, applicable=False)
        candidate, production = inputs.candidate.log_loss, inputs.production.log_loss
        return RuleResult(
            "mieux_que_production",
            label,
            passed=candidate <= production - epsilon,
            observed=f"{candidate:.4f} contre {production:.4f} (v{inputs.production_version})",
            threshold=threshold,
        )

    def _no_segment_regression(self, inputs: GateInputs) -> RuleResult:
        policy = self.policy
        label = f"pas de régression par {policy.segment_dimension}"
        threshold = f"≤ +{policy.max_segment_regression:g} de log loss ({policy.min_segment_rows} matchs min.)"
        if inputs.production is None:
            return RuleResult("segments", label, True, inputs.production_note, threshold, applicable=False)
        gaps = {
            segment: (value - inputs.production.segments[segment][0], rows)
            for segment, (value, rows) in inputs.candidate.segments.items()
            if segment in inputs.production.segments and rows >= policy.min_segment_rows
        }
        if not gaps:
            return RuleResult("segments", label, True, "aucun segment assez grand", threshold, applicable=False)
        worst, (gap, rows) = max(gaps.items(), key=lambda item: item[1][0])
        return RuleResult(
            "segments",
            label,
            passed=gap <= policy.max_segment_regression,
            observed=f"pire écart : {worst} {gap:+.4f} ({rows} matchs)",
            threshold=threshold,
        )

    def _no_strong_drift(self, data: DataEvidence) -> RuleResult:
        label = "pas de dérive forte des données"
        threshold = f"PSI max ≤ {self.policy.max_psi:g}"
        if data.max_psi is None:
            return RuleResult("derive", label, True, "première campagne : pas de référence", threshold,
                              applicable=False)  # fmt: skip
        return RuleResult(
            "derive",
            label,
            passed=data.max_psi <= self.policy.max_psi,
            observed=f"PSI max {data.max_psi:.3f} contre la campagne précédente",
            threshold=threshold,
        )
