"""Stratégie de tagging du registre MLflow : alias mobiles, tags de version, cycle de vie (détail : docs/TAGGING.md).

- **Alias** = une étiquette qui désigne UNE version à la fois et qu'on déplace : l'API charge
  `models:/atp-winner@production` et ne contient aucun numéro de version. Promouvoir ou revenir en arrière =
  déplacer l'alias, sans redéploiement.
- **Tags de version** = ce qu'on sait de chaque version : provenance, qualité, décision, cycle de vie. Les tags de
  provenance et de qualité sont posés à l'enregistrement et ne bougent plus ; la décision et le cycle de vie
  évoluent (promotion, rejet, rollback) et gardent la trace de chaque changement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum

from tennis_mlops.features import FEATURES_VERSION
from tennis_mlops.training.config import TrainConfig
from tennis_mlops.training.dataset import Dataset
from tennis_mlops.training.split import Split
from tennis_mlops.validation.suites import SUITES_VERSION

MAX_TAG_LENGTH = 5000


class Alias(StrEnum):
    """Les étiquettes mobiles du registre."""

    PRODUCTION = "production"  # servie par l'API ; déplacée par la règle de mise en service ou à la main
    PREVIOUS = "previous"  # la version remplacée par la dernière promotion : cible de `make rollback`
    STAGING = "staging"  # dernier candidat refusé, gardé en observation (scoring « shadow » en phase 7)


class Lifecycle(StrEnum):
    """Où en est une version aujourd'hui (tag `lifecycle`)."""

    CANDIDATE = "candidate"  # enregistrée, décision pas encore prise (ou campagne interrompue avant)
    PRODUCTION = "production"  # servie
    ARCHIVED = "archived"  # a été servie, remplacée par une meilleure ; reste disponible pour un rollback
    REJECTED = "rejected"  # refusée par la règle de mise en service
    ROLLED_BACK = "rolled_back"  # retirée de la production par un rollback (raison dans `rollback_reason`)


class GateStatus(StrEnum):
    """Verdict de la règle de mise en service (tag `gate_status`)."""

    PASSED = "passed"
    FAILED = "failed"


class DecidedBy(StrEnum):
    """Qui a déplacé l'alias (tag `decided_by`)."""

    PIPELINE = "pipeline"  # la campagne d'entraînement, en appliquant configs/promotion.yaml
    MANUAL = "manual"  # une personne : `make promote` ou `make rollback`, avec une raison obligatoire


# Groupes de tags d'une version, et la question à laquelle chacun répond (repris dans docs/TAGGING.md).
TAG_GROUPS: dict[str, tuple[str, ...]] = {
    "provenance": (
        "git_sha", "git_dirty", "uv_lock_sha256", "dataset_version", "data_window_start", "data_window_end",
        "simulated_now", "features_version", "config_hash", "campaign_run_id", "prefect_flow_run_id",
        "prefect_flow_run_url", "last_valid_lot",
    ),
    "qualite": (
        "data_validation", "gx_suite_version", "lots_valid", "lots_quarantined", "n_train", "n_validation", "n_test",
        "test_window", "candidate", "selection_metric",
    ),
    "decision": ("gate_status", "gate_reason", "compared_to_version", "delta_log_loss", "decided_by", "decided_at"),
    "cycle_de_vie": (
        "lifecycle", "promoted_at", "replaced_version", "archived_at", "replaced_by", "promotion_reason",
        "rollback_reason", "rolled_back_at", "rolled_back_to", "restored_at", "restored_from",
    ),
}  # fmt: skip


def now_iso() -> str:
    """Horodatage UTC lisible, à la seconde."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def config_hash(config: TrainConfig) -> str:
    """Empreinte de la configuration résolue : deux versions au même hash ont été produites avec les mêmes réglages."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def clean(tags: Mapping[str, object]) -> dict[str, str]:
    """Valeurs en texte, `None` → chaîne vide, longueur limitée à ce que MLflow accepte."""
    return {k: ("" if v is None else str(v))[:MAX_TAG_LENGTH] for k, v in tags.items()}


def version_tags(
    *,
    config: TrainConfig,
    dataset: Dataset,
    split: Split,
    context: Mapping[str, str],
    campaign_run_id: str,
    candidate: str,
    validation_score: float,
    test_score: float,
) -> dict[str, str]:
    """Tags posés à l'enregistrement : provenance + qualité, et `lifecycle=candidate` (décision à suivre)."""
    lots, metric = dataset.lots, config.selection_metric
    return clean(
        {
            # provenance : avec quoi ce modèle a-t-il été produit ?
            "git_sha": context.get("git.commit", "inconnu"),
            "git_dirty": context.get("git.dirty", "inconnu"),
            "uv_lock_sha256": context.get("uv_lock.sha256", "inconnu"),
            "dataset_version": ",".join(dataset.dataset_versions),
            "data_window_start": split.train_start.isoformat(),
            "data_window_end": dataset.last_date.isoformat(),
            "simulated_now": lots.get("watermark"),
            "features_version": FEATURES_VERSION,
            "config_hash": config_hash(config),
            "campaign_run_id": campaign_run_id,
            "prefect_flow_run_id": context.get("prefect.flow_run_id"),
            "prefect_flow_run_url": context.get("prefect.flow_run_url"),
            "last_valid_lot": lots.get("last_valid_lot"),
            # qualité : les données étaient-elles saines et suffisantes ?
            "data_validation": "passed",  # la base ne contient que des lots acceptés par la suite bloquante
            "gx_suite_version": SUITES_VERSION,
            "lots_valid": lots.get("lots_valid"),
            "lots_quarantined": lots.get("lots_quarantined"),
            "n_train": len(split.train),
            "n_validation": len(split.validation),
            "n_test": len(split.test),
            "test_window": f"{split.test_start.isoformat()} → {split.end.isoformat()}",
            "candidate": candidate,
            "selection_metric": metric,
            f"validation.{metric}": f"{validation_score:.5f}",
            f"test.{metric}": f"{test_score:.5f}",
            # cycle de vie : la décision suit (règle de mise en service)
            "lifecycle": Lifecycle.CANDIDATE.value,
        }
    )
