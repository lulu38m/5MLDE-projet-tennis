import math
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from tennis_mlops.registry.gate import (
    DEFAULT_PROMOTION_CONFIG_PATH,
    DataEvidence,
    Gate,
    GateDecision,
    GateInputs,
    ModelScores,
    PromotionPolicy,
    RuleResult,
)
from tennis_mlops.registry.tagging import GateStatus

ELO = ModelScores(log_loss=0.612, accuracy=0.674, rows=1400)
RANK = ModelScores(log_loss=0.622, accuracy=0.650, rows=1400)
DATA = DataEvidence(lots_valid=120, lots_quarantined=2, suite_version="1.0", max_psi=0.04)


def scores(log_loss: float, accuracy: float = 0.68, rows: int = 1400, **segments: tuple[float, int]) -> ModelScores:
    return ModelScores(log_loss=log_loss, accuracy=accuracy, rows=rows, segments=segments)


def inputs(candidate: ModelScores, production: ModelScores | None = None, data: DataEvidence = DATA) -> GateInputs:
    return GateInputs(candidate=candidate, baselines={"elo": ELO, "rang": RANK}, data=data, production=production,
                      production_version="3" if production else None,
                      production_note="v3" if production else "aucune version en production")  # fmt: skip


def rule(decision: GateDecision, name: str) -> RuleResult:
    return next(r for r in decision.rules if r.rule == name)


def test_first_model_is_promoted_when_it_beats_the_baselines() -> None:
    decision = Gate(PromotionPolicy()).decide(inputs(scores(0.600)))
    assert decision.passed and decision.status is GateStatus.PASSED
    assert decision.compared_to_version is None and decision.delta_log_loss is None
    assert "première mise en service" in decision.reason
    not_applicable = [r.rule for r in decision.rules if not r.applicable]
    assert not_applicable == ["mieux_que_production", "segments"]
    assert decision.tags() == {"gate_status": "passed", "gate_reason": decision.reason, "compared_to_version": "",
                               "delta_log_loss": ""}  # fmt: skip


def test_a_model_worse_than_the_formulas_is_refused() -> None:
    decision = Gate(PromotionPolicy()).decide(inputs(scores(0.693, accuracy=0.52)))
    assert decision.status is GateStatus.FAILED
    assert [r.rule for r in decision.failed_rules] == ["bat_elo", "accuracy_classement"]
    assert "Elo pur" in decision.reason and "0.6930 contre 0.6120" in decision.reason


def test_comparison_with_production_on_the_same_test() -> None:
    production = scores(0.6050, Clay=(0.60, 500), Hard=(0.61, 800))
    better = Gate(PromotionPolicy()).decide(inputs(scores(0.6020, Clay=(0.59, 500), Hard=(0.61, 800)), production))
    assert better.passed and better.compared_to_version == "3"
    assert better.delta_log_loss == pytest.approx(-0.003)
    assert better.tags()["delta_log_loss"] == "-0.00300"
    assert "contre v3" in better.reason

    worse = Gate(PromotionPolicy()).decide(inputs(scores(0.6060), production))
    assert [r.rule for r in worse.failed_rules] == ["mieux_que_production"]

    tie = Gate(PromotionPolicy()).decide(inputs(scores(0.6050), production))
    assert tie.passed  # à égalité, le plus récent (données plus fraîches) l'emporte

    strict = Gate(PromotionPolicy(min_improvement=0.005)).decide(inputs(scores(0.6020), production))
    assert [r.rule for r in strict.failed_rules] == ["mieux_que_production"]
    assert rule(strict, "mieux_que_production").threshold == "≤ production - 0.005"


def test_a_surface_must_not_degrade_even_if_the_average_improves() -> None:
    production = scores(0.6100, Clay=(0.600, 500), Grass=(0.580, 150), Carpet=(0.500, 20))
    candidate = scores(0.6000, Clay=(0.580, 500), Grass=(0.640, 150), Carpet=(0.900, 20))
    decision = Gate(PromotionPolicy()).decide(inputs(candidate, production))
    (failed,) = decision.failed_rules
    assert failed.rule == "segments"
    assert failed.observed == "pire écart : Grass +0.0600 (150 matchs)"  # Carpet (20 matchs) n'est pas jugé

    tolerant = Gate(PromotionPolicy(max_segment_regression=0.1)).decide(inputs(candidate, production))
    assert tolerant.passed

    tiny = Gate(PromotionPolicy()).decide(inputs(scores(0.60, Carpet=(0.9, 20)), scores(0.61, Carpet=(0.5, 20))))
    assert rule(tiny, "segments").observed == "aucun segment assez grand"


def test_small_test_set_and_strong_drift_are_refused() -> None:
    small = Gate(PromotionPolicy()).decide(inputs(scores(0.60, rows=200)))
    assert [r.rule for r in small.failed_rules] == ["taille_test"]

    drifted = DataEvidence(lots_valid=120, lots_quarantined=0, suite_version="1.0", max_psi=0.31)
    decision = Gate(PromotionPolicy()).decide(inputs(scores(0.60), data=drifted))
    assert [r.rule for r in decision.failed_rules] == ["derive"]
    assert "PSI max 0.310" in decision.reason

    first = DataEvidence(lots_valid=10, lots_quarantined=0, suite_version="1.0", max_psi=None)
    assert not rule(Gate(PromotionPolicy()).decide(inputs(scores(0.6), data=first)), "derive").applicable


def test_decision_table_and_json() -> None:
    decision = Gate(PromotionPolicy()).decide(inputs(scores(0.693, accuracy=0.52)))
    table = decision.table()
    assert list(table.columns) == ["règle", "résultat", "mesuré", "seuil"]
    assert table["résultat"].tolist() == ["OK", "OK", "ÉCHEC", "ÉCHEC", "sans objet", "sans objet", "OK"]
    data = decision.as_dict()
    assert data["status"] == "failed" and len(data["rules"]) == 7


def test_scores_from_the_evaluator_segment_table() -> None:
    segments = pd.DataFrame(
        {
            "dimension": ["surface", "surface", "surface", "niveau"],
            "segment": ["Clay", "Grass", "Carpet", "Grand Chelem"],
            "rows": [500, 150, 12, 300],
            "log_loss": [0.59, 0.62, math.nan, 0.55],
        }
    )
    built = ModelScores.from_metrics({"log_loss": 0.6, "accuracy": 0.7}, 1400, segments, "surface")
    assert built.segments == {"Clay": (0.59, 500), "Grass": (0.62, 150)}
    assert ModelScores.from_metrics({"log_loss": 0.6, "accuracy": 0.7}, 10).segments == {}


def test_policy_file(tmp_path: Path) -> None:
    policy = PromotionPolicy.from_yaml(DEFAULT_PROMOTION_CONFIG_PATH)
    assert policy.min_improvement == 0.0 and policy.segment_dimension == "surface"
    (tmp_path / "bad.yaml").write_text("min_test_rows: 10\nseuil_inconnu: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="seuil_inconnu"):
        PromotionPolicy.from_yaml(tmp_path / "bad.yaml")
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    assert PromotionPolicy.from_yaml(tmp_path / "empty.yaml") == PromotionPolicy()
