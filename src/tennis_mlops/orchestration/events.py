"""Journal des événements du pipeline (table `pipeline_events`) : la chronologie que lisent Grafana et les alertes.

Une ligne par fait marquant : ingestion terminée, lot mis en quarantaine, ingestion dégradée, réentraînement
demandé, version promue ou refusée (fin d'une campagne), rollback, alerte traitée. Chaque ligne garde ses identifiants
(run Prefect, lot, version de modèle) : on remonte de l'annotation Grafana au run qui l'a produite.

Écrire un événement ne doit jamais faire échouer un flow : `record` avale les erreurs de base et les signale.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)
metadata = sa.MetaData()

events = sa.Table(
    "pipeline_events",
    metadata,
    sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
    sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, index=True),
    sa.Column("kind", sa.String(32), nullable=False, index=True),
    sa.Column("severity", sa.String(16), nullable=False),
    sa.Column("title", sa.String(200), nullable=False),
    sa.Column("detail", sa.Text(), nullable=False, default=""),
    sa.Column("simulated_date", sa.Date()),
    sa.Column("flow_run_id", sa.String(64), index=True),
    sa.Column("lot_id", sa.String(80)),
    sa.Column("model_version", sa.String(16)),
    sa.Column("ref", sa.String(64), index=True),  # identifiant externe (événement Prefect) : dédoublonnage
    sa.Column("payload", sa.Text(), nullable=False, default="{}"),
)


class EventKind(StrEnum):
    """Les faits enregistrés."""

    INGESTION_COMPLETED = "ingestion_completed"
    LOT_QUARANTINED = "lot_quarantined"
    INGESTION_DEGRADED = "ingestion_degraded"
    TRAINING_REQUESTED = "training_requested"
    MODEL_PROMOTED = "model_promoted"
    MODEL_REJECTED = "model_rejected"
    MANUAL_PROMOTION = "manual_promotion"
    ROLLBACK = "rollback"
    ALERT = "alert"


class Severity(StrEnum):
    """Gravité, pour la couleur des annotations."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    FAILURE = "failure"


@dataclass(frozen=True)
class PipelineEvent:
    """Un événement à enregistrer."""

    kind: EventKind
    severity: Severity
    title: str
    detail: str = ""
    simulated_date: date | None = None
    flow_run_id: str | None = None
    lot_id: str | None = None
    model_version: str | None = None
    ref: str | None = None
    payload: Mapping[str, Any] | None = None


class PipelineEvents:
    """Lecture et écriture de la table `pipeline_events` (PostgreSQL de la stack, SQLite dans les tests)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def from_url(cls, url: str) -> PipelineEvents:
        """Ouvre la base du projet."""
        return cls(sa.create_engine(url, pool_pre_ping=True))

    def create_schema(self) -> None:
        """Crée la table si elle manque (idempotent)."""
        metadata.create_all(self._engine)

    def record(self, event: PipelineEvent, at: datetime | None = None) -> bool:
        """Enregistre l'événement ; False (et un avertissement) si la base est indisponible, sans lever."""
        row = {
            "occurred_at": at or datetime.now(UTC),
            "kind": event.kind.value,
            "severity": event.severity.value,
            "title": event.title[:200],
            "detail": event.detail,
            "simulated_date": event.simulated_date,
            "flow_run_id": event.flow_run_id,
            "lot_id": event.lot_id,
            "model_version": event.model_version,
            "ref": event.ref,
            "payload": json.dumps(dict(event.payload or {}), ensure_ascii=False, default=str),
        }
        try:
            self.create_schema()
            with self._engine.begin() as conn:
                conn.execute(sa.insert(events).values(**row))
        except SQLAlchemyError as exc:
            logger.warning("événement non enregistré (%s) : %s", event.kind.value, exc)
            return False
        return True

    def recent(
        self, limit: int = 50, kinds: tuple[EventKind, ...] = (), flow_run_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Les derniers événements (d'un type, d'un flow run), du plus récent au plus ancien."""
        query = sa.select(events).order_by(events.c.occurred_at.desc(), events.c.id.desc()).limit(limit)
        if kinds:
            query = query.where(events.c.kind.in_([k.value for k in kinds]))
        if flow_run_id:
            query = query.where(events.c.flow_run_id == flow_run_id)
        self.create_schema()
        with self._engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(query).mappings()]
        for row in rows:
            row["payload"] = json.loads(row["payload"] or "{}")
        return rows

    def last(self, *kinds: EventKind) -> dict[str, Any] | None:
        """Le dernier événement d'un (ou plusieurs) type(s)."""
        found = self.recent(limit=1, kinds=kinds)
        return found[0] if found else None

    def seen(self, ref: str, kind: EventKind = EventKind.ALERT) -> bool:
        """Vrai si un événement de ce type porte déjà cette référence (une alerte n'est traitée qu'une fois)."""
        self.create_schema()
        query = sa.select(sa.func.count()).select_from(events).where(events.c.kind == kind.value, events.c.ref == ref)
        with self._engine.connect() as conn:
            return int(conn.execute(query).scalar_one()) > 0
