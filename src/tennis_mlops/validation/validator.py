"""Exécution des suites sur un lot, comme dans les TP : contexte GX sur disque, checkpoints, Data Docs HTML.

Le contexte vit dans `state/gx/` (hors Git). Chaque lot met à jour les deux suites (leurs paramètres dépendent du lot :
date simulée, volume attendu, plage historique) puis passe par deux checkpoints qui alimentent les Data Docs : la page
HTML officielle de GX, consultable pour chaque validation.

Toute erreur technique de GX (colonne absente, type inattendu…) compte comme un échec : un contrôle qui n'a pas pu
s'exécuter ne peut pas laisser passer un lot.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")  # aucune télémétrie

import great_expectations as gx
from great_expectations.checkpoint import UpdateDataDocsAction
from great_expectations.core.batch_definition import BatchDefinition
from great_expectations.data_context import AbstractDataContext
from great_expectations.data_context.types.base import ProgressBarsConfig
from great_expectations.expectations.expectation import Expectation

import tennis_mlops.validation.expectations  # noqa: F401 — enregistre les métriques custom auprès de GX
from tennis_mlops.ingestion.lots import Lot
from tennis_mlops.validation.suites import (
    BLOCKING_SUITE,
    MONITORING_SUITE,
    SUITES_VERSION,
    LotExpectations,
    RankGapReference,
    blocking_expectations,
    monitoring_expectations,
)

DATA_DOCS_INDEX = Path("gx") / "uncommitted" / "data_docs" / "local_site" / "index.html"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Résultat d'une expectation."""

    expectation: str
    target: str
    success: bool
    blocking: bool
    unexpected_count: int | None = None
    sample: list[str] = field(default_factory=list)
    error: str | None = None

    def describe(self) -> str:
        """Une ligne lisible pour les logs et les alertes."""
        detail = self.error or (f"{self.unexpected_count} ligne(s), ex. {self.sample[:3]}" if self.sample else "")
        return f"{self.expectation}({self.target}) {detail}".strip()


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Verdict d'un lot : `success` ne dépend que de la suite bloquante."""

    lot_id: str
    success: bool
    checks: list[CheckResult]
    docs_index: str | None
    suites_version: str = SUITES_VERSION

    @property
    def failures(self) -> list[CheckResult]:
        """Échecs bloquants."""
        return [c for c in self.checks if c.blocking and not c.success]

    @property
    def warnings(self) -> list[CheckResult]:
        """Écarts de surveillance (non bloquants)."""
        return [c for c in self.checks if not c.blocking and not c.success]

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON (archivée avec le lot)."""
        return asdict(self)

    def to_json(self) -> str:
        """JSON indenté."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)


class LotValidator:
    """Valide les lots avec Great Expectations (un contexte réutilisé pour tous les lots d'une exécution)."""

    def __init__(self, gx_dir: Path) -> None:
        self._gx_dir = gx_dir
        self._context: AbstractDataContext | None = None
        self._batch: BatchDefinition[Any] | None = None

    @property
    def docs_index(self) -> Path:
        """Page d'accueil des Data Docs."""
        return self._gx_dir / DATA_DOCS_INDEX

    def validate(self, lot: Lot, reference: RankGapReference | None = None) -> ValidationReport:
        """Exécute la suite bloquante puis la suite de surveillance sur le lot."""
        params = LotExpectations(simulated_now=lot.simulated_now, expected_rows=lot.expected_rows)
        checks = self._run(BLOCKING_SUITE, blocking_expectations(params), lot, blocking=True)
        monitoring = monitoring_expectations(reference)
        if monitoring:
            checks += self._run(MONITORING_SUITE, monitoring, lot, blocking=False)
        success = all(c.success for c in checks if c.blocking)
        docs = str(self.docs_index) if self.docs_index.exists() else None
        return ValidationReport(lot_id=lot.lot_id, success=success, checks=checks, docs_index=docs)

    # ------------------------------------------------------------------ GX

    def _ensure_context(self) -> tuple[AbstractDataContext, BatchDefinition[Any]]:
        if self._context is None or self._batch is None:
            self._gx_dir.mkdir(parents=True, exist_ok=True)
            context = gx.get_context(mode="file", project_root_dir=self._gx_dir)
            context.variables.progress_bars = ProgressBarsConfig(globally=False)
            source = context.data_sources.add_or_update_pandas("lots")
            asset = next((a for a in source.assets if a.name == "lot"), None) or source.add_dataframe_asset("lot")
            batch = next((b for b in asset.batch_definitions if b.name == "complet"), None)
            self._batch = batch or asset.add_batch_definition_whole_dataframe("complet")
            self._context = context
        return self._context, self._batch

    def _run(self, suite_name: str, rules: list[Expectation], lot: Lot, blocking: bool) -> list[CheckResult]:
        try:
            context, batch = self._ensure_context()
            suite = gx.ExpectationSuite(name=suite_name, meta={"version": SUITES_VERSION, "lot": lot.lot_id})
            for rule in rules:
                suite.add_expectation(rule)
            suite = context.suites.add_or_update(suite)
            definition = context.validation_definitions.add_or_update(
                gx.ValidationDefinition(name=suite_name, data=batch, suite=suite)
            )
            checkpoint = context.checkpoints.add_or_update(
                gx.Checkpoint(
                    name=suite_name,
                    validation_definitions=[definition],
                    actions=[UpdateDataDocsAction(name="data_docs")],
                )
            )
            run = checkpoint.run(batch_parameters={"dataframe": lot.frame})
        except Exception as exc:
            return [CheckResult(suite_name, lot.lot_id, False, blocking, error=f"{type(exc).__name__}: {exc}")]
        results: list[CheckResult] = []
        for validation in run.run_results.values():
            for result in validation.results:
                results.append(_check(result.to_json_dict(), blocking))
        return results


def _check(result: dict[str, Any], blocking: bool) -> CheckResult:
    config = result.get("expectation_config", {})
    kwargs = config.get("kwargs", {})
    target = kwargs.get("column") or "/".join(str(kwargs[k]) for k in ("column_A", "column_B") if k in kwargs)
    detail = result.get("result") or {}
    exception = result.get("exception_info") or {}
    error = None
    if exception.get("raised_exception"):
        error = str(exception.get("exception_message") or "erreur GX")[:300]
    sample = [str(v) for v in (detail.get("partial_unexpected_list") or [])[:5]]
    if not sample and "observed_value" in detail and not result.get("success"):
        sample = [f"observé : {detail['observed_value']}"]
    return CheckResult(
        expectation=str(config.get("type", "inconnue")),
        target=str(target or "table"),
        success=bool(result.get("success")),
        blocking=blocking,
        unexpected_count=detail.get("unexpected_count"),
        sample=sample,
        error=error,
    )
