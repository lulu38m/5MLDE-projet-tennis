"""Stockage des lots : PostgreSQL en exécution réelle, SQLite dans les tests (SQLAlchemy Core, SQL portable).

| Table | Contenu |
|---|---|
| `matches` | un match validé par ligne ; clé `match_id` (la clé de feedback) |
| `ingestion_lots` | un lot par ligne : statut (`valid` / `quarantined`), volumes, contrôles en échec, rapport GX |
| `ingestion_state` | le watermark : date simulée jusqu'à laquelle tout est ingéré |

Un lot validé est écrit en **une transaction** (matchs + ligne de lot) : jamais de lot à moitié stocké. Réécrire un lot
déjà stocké n'ajoute rien (idempotence par `match_id`), ce qui rend les reprises sans risque.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from tennis_mlops.ingestion.lots import PLAYER_FIELDS, Lot
from tennis_mlops.validation.suites import RankGapReference
from tennis_mlops.validation.validator import ValidationReport

WATERMARK_KEY = "results_watermark"

metadata = sa.MetaData()


def _player_columns(side: str) -> list[sa.Column[Any]]:
    types: dict[str, sa.types.TypeEngine[Any]] = {
        "id": sa.BigInteger(),
        "name": sa.String(120),
        "hand": sa.String(4),
        "ht": sa.Integer(),
        "ioc": sa.String(8),
        "age": sa.Float(),
        "rank": sa.Integer(),
        "rank_points": sa.Integer(),
        "seed": sa.Integer(),
        "entry": sa.String(8),
    }
    return [sa.Column(f"{side}_{f}", types[f], nullable=f != "id") for f in PLAYER_FIELDS]


matches = sa.Table(
    "matches",
    metadata,
    sa.Column("match_id", sa.String(64), primary_key=True),
    # Jusqu'à 35 caractères dans la source (Coupe Davis : « 2016-M-DC-2016-G1-AM-M-BAR-ECU-01 »).
    sa.Column("tourney_id", sa.String(64), nullable=False, index=True),
    sa.Column("tourney_name", sa.String(120)),
    sa.Column("tourney_level", sa.String(4), index=True),
    sa.Column("surface", sa.String(16)),
    sa.Column("draw_size", sa.Integer()),
    sa.Column("best_of", sa.Integer()),
    sa.Column("round", sa.String(8)),
    sa.Column("match_num", sa.Integer()),
    sa.Column("match_date", sa.Date(), nullable=False, index=True),
    sa.Column("status", sa.String(16)),
    sa.Column("score", sa.String(64)),
    sa.Column("minutes", sa.Integer()),
    *_player_columns("winner"),
    *_player_columns("loser"),
    sa.Column("lot_id", sa.String(80), nullable=False),
    sa.Column("dataset_version", sa.String(32)),
    sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
)

lots = sa.Table(
    "ingestion_lots",
    metadata,
    sa.Column("lot_id", sa.String(80), primary_key=True),
    sa.Column("kind", sa.String(16), nullable=False),
    sa.Column("tourney_id", sa.String(64)),
    sa.Column("tourney_level", sa.String(4)),
    sa.Column("status", sa.String(16), nullable=False, index=True),
    sa.Column("rows_received", sa.Integer(), nullable=False),
    sa.Column("rows_expected", sa.Integer()),
    sa.Column("rows_inserted", sa.Integer(), nullable=False),
    sa.Column("end_date", sa.Date()),
    sa.Column("simulated_now", sa.Date(), nullable=False),
    sa.Column("dataset_version", sa.String(32)),
    sa.Column("failed_checks", sa.Text(), nullable=False),
    sa.Column("warnings", sa.Text(), nullable=False),
    sa.Column("archive_path", sa.Text()),
    sa.Column("docs_index", sa.Text()),
    sa.Column("flow_run_id", sa.String(64)),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

state = sa.Table(
    "ingestion_state",
    metadata,
    sa.Column("key", sa.String(64), primary_key=True),
    sa.Column("value", sa.Text(), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

MATCH_COLUMNS = tuple(c.name for c in matches.columns if c.name not in {"lot_id", "dataset_version", "ingested_at"})


class LotStatus:
    """Statuts possibles d'un lot."""

    VALID = "valid"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class LotRecord:
    """Trace d'un lot, telle que stockée."""

    lot_id: str
    status: str
    rows_inserted: int


def _clean(value: object) -> object:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):  # scalaires numpy
        return value.item()
    return value


def _rows(frame: pd.DataFrame, lot: Lot, now: datetime) -> list[dict[str, Any]]:
    present = [c for c in MATCH_COLUMNS if c in frame.columns]
    records = []
    for raw in frame[present].to_dict("records"):
        row: dict[str, Any] = {str(k): _clean(v) for k, v in raw.items()}
        row["match_date"] = date.fromisoformat(str(row["match_date"]))
        for key in ("minutes", "winner_ht", "loser_ht", "winner_rank", "loser_rank", "winner_seed", "loser_seed"):
            if row.get(key) is not None:
                row[key] = int(row[key])
        row.update(lot_id=lot.lot_id, dataset_version=lot.payload.dataset_version, ingested_at=now)
        records.append(row)
    return records


class IngestionStore:
    """Lecture et écriture des données ingérées."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def from_url(cls, url: str) -> IngestionStore:
        """Ouvre la base (`postgresql+psycopg://…` ou `sqlite:///…`)."""
        return cls(sa.create_engine(url, pool_pre_ping=True))

    @property
    def engine(self) -> Engine:
        """Le moteur SQLAlchemy (partagé avec le journal des événements, même base)."""
        return self._engine

    def create_schema(self) -> None:
        """Crée les tables manquantes et élargit les colonnes devenues trop courtes (idempotent)."""
        metadata.create_all(self._engine)
        for table, column, length in self.widenings():
            if self._engine.dialect.name == "postgresql":  # SQLite ne contrôle pas la longueur des VARCHAR
                with self._engine.begin() as conn:
                    conn.execute(sa.text(f'ALTER TABLE "{table}" ALTER COLUMN "{column}" TYPE VARCHAR({length})'))

    def widenings(self) -> list[tuple[str, str, int]]:
        """Colonnes texte plus courtes en base que dans le schéma déclaré : (table, colonne, nouvelle longueur).

        Élargir un VARCHAR ne perd aucune donnée : c'est la seule migration faite automatiquement (une base créée
        en phase 3 avait `tourney_id` en 32 caractères, trop court pour la Coupe Davis).
        """
        inspector = sa.inspect(self._engine)
        plan = []
        for table in metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            current = {c["name"]: getattr(c["type"], "length", None) for c in inspector.get_columns(table.name)}
            for column in table.columns:
                wanted = getattr(column.type, "length", None)
                have = current.get(column.name)
                if wanted and have and have < wanted:
                    plan.append((table.name, column.name, int(wanted)))
        return plan

    # ------------------------------------------------------------------ watermark

    def watermark(self, default: date) -> date:
        """Date simulée jusqu'à laquelle tout est ingéré (exclue)."""
        with self._engine.connect() as conn:
            value = conn.execute(sa.select(state.c.value).where(state.c.key == WATERMARK_KEY)).scalar_one_or_none()
        return date.fromisoformat(value) if value else default

    def set_watermark(self, value: date) -> None:
        """Avance (ou recule) le watermark."""
        with self._engine.begin() as conn:
            conn.execute(sa.delete(state).where(state.c.key == WATERMARK_KEY))
            conn.execute(sa.insert(state).values(key=WATERMARK_KEY, value=value.isoformat(), updated_at=_now()))

    # ------------------------------------------------------------------ lots

    def valid_lot_ids(self) -> set[str]:
        """Lots déjà stockés avec succès (à ne pas refaire)."""
        with self._engine.connect() as conn:
            return set(conn.execute(sa.select(lots.c.lot_id).where(lots.c.status == LotStatus.VALID)).scalars())

    def save_valid_lot(self, lot: Lot, report: ValidationReport, archive: str | None, flow_run_id: str | None) -> int:
        """Écrit les nouveaux matchs du lot et sa trace, en une transaction ; renvoie le nombre de matchs ajoutés."""
        now = _now()
        rows = _rows(lot.frame, lot, now)
        with self._engine.begin() as conn:
            known = self._existing_ids(conn, [str(r["match_id"]) for r in rows])
            new_rows = [r for r in rows if r["match_id"] not in known]
            if new_rows:
                conn.execute(sa.insert(matches), new_rows)
            self._upsert_lot(conn, lot, report, LotStatus.VALID, len(new_rows), archive, flow_run_id, now)
        return len(new_rows)

    def save_quarantined_lot(
        self, lot: Lot, report: ValidationReport, archive: str | None, flow_run_id: str | None
    ) -> None:
        """Trace un lot refusé (aucun match n'est stocké)."""
        with self._engine.begin() as conn:
            self._upsert_lot(conn, lot, report, LotStatus.QUARANTINED, 0, archive, flow_run_id, _now())

    def lot(self, lot_id: str) -> Mapping[str, Any] | None:
        """Trace d'un lot."""
        with self._engine.connect() as conn:
            row = conn.execute(sa.select(lots).where(lots.c.lot_id == lot_id)).mappings().one_or_none()
        return dict(row) if row else None

    def count_matches(self) -> int:
        """Nombre de matchs stockés."""
        with self._engine.connect() as conn:
            return int(conn.execute(sa.select(sa.func.count()).select_from(matches)).scalar_one())

    def fingerprint(self) -> str:
        """Empreinte des matchs stockés (nombre, dernier ajout, dernier match) : clé du cache du jeu d'entraînement.

        Les lots ne font qu'ajouter des matchs : tant que l'empreinte ne change pas, le jeu de features est le même.
        """
        query = sa.select(sa.func.count(), sa.func.max(matches.c.ingested_at), sa.func.max(matches.c.match_date))
        with self._engine.connect() as conn:
            count, last_ingested, last_match = conn.execute(query).one()
        raw = f"{count}|{last_ingested}|{last_match}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def count_matches_ingested_after(self, since: datetime | None) -> int:
        """Matchs stockés après un instant (tous si `since` est None) : de quoi décider un réentraînement."""
        query = sa.select(sa.func.count()).select_from(matches)
        if since is not None:
            query = query.where(matches.c.ingested_at > since)
        with self._engine.connect() as conn:
            return int(conn.execute(query).scalar_one())

    def matches_frame(self) -> pd.DataFrame:
        """Tous les matchs validés, pour l'entraînement (quelques Mo : 47 000 matchs au plus)."""
        with self._engine.connect() as conn:
            return pd.DataFrame(conn.execute(sa.select(matches)).mappings().all(), columns=list(matches.columns.keys()))

    def lots_summary(self) -> dict[str, Any]:
        """Provenance du jeu d'entraînement : lots valides / en quarantaine, dernier lot validé, watermark."""
        with self._engine.connect() as conn:
            grouped = conn.execute(sa.select(lots.c.status, sa.func.count()).group_by(lots.c.status)).all()
            counts = {str(status): int(n) for status, n in grouped}
            last = conn.execute(
                sa.select(lots.c.lot_id, lots.c.end_date, lots.c.docs_index)
                .where(lots.c.status == LotStatus.VALID)
                .order_by(lots.c.end_date.desc().nulls_last(), lots.c.updated_at.desc())
                .limit(1)
            ).one_or_none()
            watermark = conn.execute(sa.select(state.c.value).where(state.c.key == WATERMARK_KEY)).scalar_one_or_none()
            refused = conn.execute(
                sa.select(lots.c.lot_id).where(lots.c.status == LotStatus.QUARANTINED).order_by(lots.c.lot_id).limit(10)
            ).scalars()
            quarantined_ids = [str(lot_id) for lot_id in refused]
        return {
            "lots_valid": int(counts.get(LotStatus.VALID, 0)),
            "lots_quarantined": int(counts.get(LotStatus.QUARANTINED, 0)),
            "quarantined_lots": quarantined_ids,
            "last_valid_lot": last.lot_id if last else None,
            "last_valid_lot_end": last.end_date.isoformat() if last and last.end_date else None,
            "last_valid_lot_docs": last.docs_index if last else None,
            "watermark": watermark,
        }

    # ------------------------------------------------------------------ référence de dérive

    def rank_gap_reference(
        self, level: str, before: date, min_tournaments: int, quantiles: tuple[float, float]
    ) -> RankGapReference | None:
        """Plage historique de la médiane de l'écart de classement par tournoi, pour un niveau ; None si trop peu."""
        query = sa.select(matches.c.tourney_id, matches.c.winner_rank, matches.c.loser_rank).where(
            matches.c.tourney_level == level, matches.c.match_date < before
        )
        with self._engine.connect() as conn:
            frame = pd.DataFrame(conn.execute(query).all(), columns=["tourney_id", "winner_rank", "loser_rank"])
        if frame.empty:
            return None
        frame["gap"] = (frame["winner_rank"] - frame["loser_rank"]).abs()
        medians = frame.groupby("tourney_id")["gap"].median().dropna()
        if len(medians) < min_tournaments:
            return None
        low, high = (float(medians.quantile(q)) for q in quantiles)
        return RankGapReference(level=level, low=low, high=high, tournaments=len(medians))

    # ------------------------------------------------------------------ interne

    @staticmethod
    def _existing_ids(conn: Connection, ids: Iterable[str]) -> set[str]:
        wanted = list(ids)
        found: set[str] = set()
        for start in range(0, len(wanted), 500):  # requêtes bornées, même pour l'amorçage (15 000 matchs)
            chunk = wanted[start : start + 500]
            found |= set(conn.execute(sa.select(matches.c.match_id).where(matches.c.match_id.in_(chunk))).scalars())
        return found

    @staticmethod
    def _upsert_lot(
        conn: Connection,
        lot: Lot,
        report: ValidationReport,
        status: str,
        inserted: int,
        archive: str | None,
        flow_run_id: str | None,
        now: datetime,
    ) -> None:
        conn.execute(sa.delete(lots).where(lots.c.lot_id == lot.lot_id))
        conn.execute(
            sa.insert(lots).values(
                lot_id=lot.lot_id,
                kind=lot.kind.value,
                tourney_id=lot.tourney_id,
                tourney_level=lot.tourney_level,
                status=status,
                rows_received=lot.rows,
                rows_expected=lot.expected_rows,
                rows_inserted=inserted,
                end_date=lot.end_date,
                simulated_now=lot.simulated_now,
                dataset_version=lot.payload.dataset_version,
                failed_checks=json.dumps([c.describe() for c in report.failures], ensure_ascii=False),
                warnings=json.dumps([c.describe() for c in report.warnings], ensure_ascii=False),
                archive_path=archive,
                docs_index=report.docs_index,
                flow_run_id=flow_run_id,
                updated_at=now,
            )
        )


def _now() -> datetime:
    return datetime.now(UTC)
