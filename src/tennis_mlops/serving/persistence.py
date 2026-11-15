"""Traces de l'API dans PostgreSQL : chaque requête, chaque prédiction, chaque erreur, chaque résultat réel.

| Table | Une ligne par | Clé | Sert à |
|---|---|---|---|
| `api_requests` | requête HTTP | `request_id` | latence, débit, codes (dashboard technique) |
| `predictions` | match prédit | `prediction_id` (`match_id` indexé) | version, probabilité, features utilisées |
| `errors` | réponse 4xx/5xx | `id` | `422` détaillés, erreurs internes (scénario S8) |
| `feedbacks` | match dont on connaît le résultat | `match_id` | accuracy et log loss en ligne, par version |

Vues : `prediction_outcomes` (une prédiction, sa version, la durée de sa requête et le vrai résultat) et, sous
PostgreSQL, `online_metrics` (accuracy, log loss et Brier en ligne par version), lues par Grafana.

Les requêtes, prédictions et erreurs partent par un écrivain en arrière-plan (`AsyncWriter`) : la réponse n'attend
pas la base. Si la base ne suit plus, la file déborde et les lignes en trop sont comptées (`dropped`), jamais
bloquantes. Les résultats réels, eux, sont écrits tout de suite : le client attend la confirmation.

Les tables appartiennent au compte `app_writer` (créées par l'API au démarrage) ; `grafana_reader` et `tennis` les
lisent (droits posés par `postgres-setup`, voir compose.yml).
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from tennis_mlops.features.builder import PLAYED_STATUSES
from tennis_mlops.ingestion.store import matches

logger = logging.getLogger(__name__)
metadata = sa.MetaData()
EPSILON = 1e-15

api_requests = sa.Table(
    "api_requests",
    metadata,
    sa.Column("request_id", sa.String(64), primary_key=True),
    sa.Column("received_at", sa.DateTime(timezone=True), nullable=False, index=True),
    sa.Column("method", sa.String(8), nullable=False),
    sa.Column("path", sa.String(64), nullable=False, index=True),
    sa.Column("status_code", sa.Integer(), nullable=False, index=True),
    sa.Column("duration_ms", sa.Float(), nullable=False),
    sa.Column("client", sa.String(32), index=True),
    sa.Column("model_version", sa.String(16), index=True),
    sa.Column("predictions", sa.Integer(), nullable=False, default=0),
)

predictions = sa.Table(
    "predictions",
    metadata,
    sa.Column("prediction_id", sa.String(32), primary_key=True),
    sa.Column("request_id", sa.String(64), nullable=False, index=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    sa.Column("client", sa.String(32), index=True),
    sa.Column("match_id", sa.String(64), index=True),
    sa.Column("match_date", sa.Date(), nullable=False),
    sa.Column("surface", sa.String(16)),
    sa.Column("tourney_level", sa.String(4)),
    sa.Column("round", sa.String(8)),
    sa.Column("best_of", sa.Integer()),
    sa.Column("p1_id", sa.Integer(), nullable=False),
    sa.Column("p2_id", sa.Integer(), nullable=False),
    sa.Column("p1_name", sa.String(120)),
    sa.Column("p2_name", sa.String(120)),
    sa.Column("p1_win_probability", sa.Float(), nullable=False),
    sa.Column("predicted_winner_id", sa.Integer(), nullable=False),
    sa.Column("model_name", sa.String(64), nullable=False),
    sa.Column("model_version", sa.String(16), nullable=False, index=True),
    sa.Column("run_id", sa.String(64), nullable=False),
    sa.Column("features_as_of", sa.Date()),
    sa.Column("batch_size", sa.Integer(), nullable=False, default=1),
    sa.Column("features", sa.Text(), nullable=False, default="{}"),
    sa.Column("warnings", sa.Text(), nullable=False, default="[]"),
)

errors = sa.Table(
    "errors",
    metadata,
    sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
    sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, index=True),
    sa.Column("request_id", sa.String(64), index=True),
    sa.Column("method", sa.String(8)),
    sa.Column("path", sa.String(64)),
    sa.Column("status_code", sa.Integer(), nullable=False, index=True),
    sa.Column("error_type", sa.String(64), nullable=False),
    sa.Column("message", sa.Text(), nullable=False),
    sa.Column("detail", sa.Text(), nullable=False, default="null"),
    sa.Column("client", sa.String(32)),
)

feedbacks = sa.Table(
    "feedbacks",
    metadata,
    sa.Column("match_id", sa.String(64), primary_key=True),
    sa.Column("winner_id", sa.Integer(), nullable=False),
    sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, index=True),
    sa.Column("client", sa.String(32)),
    sa.Column("request_id", sa.String(64)),
)

TABLES = {"api_requests": api_requests, "predictions": predictions, "errors": errors}

PREDICTION_OUTCOMES = """
CREATE VIEW prediction_outcomes AS
SELECT p.prediction_id, p.created_at, p.client, p.match_id, p.match_date, p.surface, p.tourney_level,
       p.p1_id, p.p2_id, p.p1_name, p.p2_name, p.p1_win_probability, p.predicted_winner_id,
       p.model_name, p.model_version, p.run_id, p.batch_size,
       r.duration_ms, r.status_code,
       f.winner_id AS actual_winner_id, f.updated_at AS feedback_at,
       CASE WHEN f.winner_id IS NULL THEN NULL
            WHEN f.winner_id = p.predicted_winner_id THEN 1 ELSE 0 END AS correct,
       CASE WHEN f.winner_id IS NULL THEN NULL
            WHEN f.winner_id = p.p1_id THEN p.p1_win_probability ELSE 1 - p.p1_win_probability END AS winner_probability
FROM predictions p
LEFT JOIN api_requests r ON r.request_id = p.request_id
LEFT JOIN feedbacks f ON f.match_id = p.match_id
"""

ONLINE_METRICS = """
CREATE VIEW online_metrics AS
SELECT model_version,
       COUNT(*) AS predictions,
       COUNT(actual_winner_id) AS with_feedback,
       AVG(correct::float) AS accuracy,
       AVG(CASE WHEN winner_probability IS NOT NULL THEN -LN(GREATEST(winner_probability, 1e-15)) END) AS log_loss,
       AVG(POWER(1 - winner_probability, 2)) AS brier,
       AVG(duration_ms) AS mean_request_ms
FROM prediction_outcomes
GROUP BY model_version
"""


@dataclass(frozen=True)
class OnlineMetrics:
    """Qualité en ligne d'une version : uniquement sur les matchs dont le résultat est connu."""

    model_version: str
    predictions: int
    with_feedback: int
    accuracy: float | None
    log_loss: float | None
    brier: float | None

    def as_dict(self) -> dict[str, Any]:
        """Pour `GET /model`."""
        return {
            "predictions": self.predictions,
            "with_feedback": self.with_feedback,
            "coverage": round(self.with_feedback / self.predictions, 4) if self.predictions else None,
            "accuracy": _round(self.accuracy),
            "log_loss": _round(self.log_loss),
            "brier": _round(self.brier),
        }


class ApiStore:
    """Écritures et lectures des tables de l'API."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def from_url(cls, url: str) -> ApiStore:
        """Ouvre la base (`postgresql+psycopg://…` ou `sqlite:///…`)."""
        return cls(sa.create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
                   if url.startswith("postgresql") else sa.create_engine(url))  # fmt: skip

    @property
    def engine(self) -> Engine:
        """Le moteur SQLAlchemy."""
        return self._engine

    def create_schema(self) -> None:
        """Tables et vues (idempotent)."""
        metadata.create_all(self._engine)
        postgres = self._engine.dialect.name == "postgresql"
        with self._engine.begin() as conn:
            if postgres:  # CREATE OR REPLACE : plusieurs processus de l'API peuvent démarrer en même temps
                conn.execute(sa.text(PREDICTION_OUTCOMES.replace("CREATE VIEW", "CREATE OR REPLACE VIEW", 1)))
                conn.execute(sa.text(ONLINE_METRICS.replace("CREATE VIEW", "CREATE OR REPLACE VIEW", 1)))
            else:
                conn.execute(sa.text("DROP VIEW IF EXISTS prediction_outcomes"))
                conn.execute(sa.text(PREDICTION_OUTCOMES))

    # ------------------------------------------------------------------ écritures

    def insert(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Insère des lignes (une transaction)."""
        if rows:
            with self._engine.begin() as conn:
                conn.execute(sa.insert(TABLES[table]), list(rows))

    def record_feedback(self, match_id: str, winner_id: int, client: str | None, request_id: str) -> bool:
        """Enregistre (ou remplace) le résultat d'un match ; True s'il en remplaçait un."""
        now = datetime.now(UTC)
        with self._engine.begin() as conn:
            existed = conn.execute(sa.select(feedbacks.c.match_id).where(feedbacks.c.match_id == match_id)).first()
            values = {"match_id": match_id, "winner_id": winner_id, "received_at": now, "updated_at": now,
                      "client": client, "request_id": request_id}  # fmt: skip
            insert = (postgresql.insert if self._engine.dialect.name == "postgresql" else sqlite.insert)(feedbacks)
            update = {"winner_id": winner_id, "updated_at": now, "client": client, "request_id": request_id}
            conn.execute(insert.values(**values).on_conflict_do_update(index_elements=["match_id"], set_=update))
        return existed is not None

    # ------------------------------------------------------------------ lectures

    def predictions_for(self, match_id: str) -> list[dict[str, Any]]:
        """Les prédictions d'un match, de la plus ancienne à la plus récente."""
        query = sa.select(predictions).where(predictions.c.match_id == match_id).order_by(predictions.c.created_at)
        with self._engine.connect() as conn:
            return [dict(r) for r in conn.execute(query).mappings()]

    def predicted_match_ids(self, match_ids: Iterable[str], client: str | None = None) -> set[str]:
        """Parmi ces matchs, ceux qui ont déjà une prédiction (de ce client, si précisé)."""
        wanted = list(set(match_ids))
        if not wanted:
            return set()
        query = sa.select(predictions.c.match_id).where(predictions.c.match_id.in_(wanted)).distinct()
        if client is not None:
            query = query.where(predictions.c.client == client)
        with self._engine.connect() as conn:
            return {str(r[0]) for r in conn.execute(query)}

    def results_awaiting_feedback(self, limit: int = 1000) -> list[tuple[str, int]]:
        """Matchs prédits dont le résultat validé est en base, mais pas encore remonté : (match_id, vainqueur)."""
        query = (
            sa.select(matches.c.match_id, matches.c.winner_id)
            .where(matches.c.match_id.in_(sa.select(predictions.c.match_id).where(predictions.c.match_id.is_not(None))))
            .where(matches.c.match_id.not_in(sa.select(feedbacks.c.match_id)))
            .where(matches.c.status.in_(sorted(PLAYED_STATUSES)))
            .order_by(matches.c.match_date, matches.c.match_id)
            .limit(limit)
        )
        with self._engine.connect() as conn:
            return [(str(r[0]), int(r[1])) for r in conn.execute(query)]

    def online_metrics(self, model_version: str | None = None) -> list[OnlineMetrics]:
        """Accuracy, log loss et Brier en ligne par version (calculés ici : SQLite n'a pas `LN`)."""
        sql = "SELECT model_version, correct, winner_probability FROM prediction_outcomes"
        query = (
            sa.text(sql + " WHERE model_version = :v").bindparams(v=model_version) if model_version else sa.text(sql)
        )
        totals: dict[str, list[Any]] = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0])
        with self._engine.connect() as conn:
            for version, correct, probability in conn.execute(query):
                acc = totals[str(version)]
                acc[0] += 1
                if probability is not None:
                    acc[1] += 1
                    acc[2] += int(correct)
                    acc[3] += -math.log(max(float(probability), EPSILON))
                    acc[4] += (1 - float(probability)) ** 2
        return [
            OnlineMetrics(version, n, k, acc / k if k else None, loss / k if k else None, brier / k if k else None)
            for version, (n, k, acc, loss, brier) in sorted(totals.items(), key=lambda kv: _version_key(kv[0]))
        ]


class AsyncWriter:
    """File d'écriture en arrière-plan : la requête dépose ses lignes et repart ; un thread les insère par paquets."""

    def __init__(self, store: ApiStore, max_queue: int = 10_000, flush_s: float = 0.5, batch: int = 500) -> None:
        self.store = store
        self.flush_s = flush_s
        self.batch = batch
        self._queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.dropped = 0
        self.failed = 0
        self.written = 0

    @property
    def pending(self) -> int:
        """Lignes en attente d'écriture."""
        return self._queue.qsize()

    def put(self, table: str, row: dict[str, Any]) -> bool:
        """Dépose une ligne ; False (et compté) si la file est pleine."""
        try:
            self._queue.put_nowait((table, row))
        except queue.Full:
            with self._lock:
                self.dropped += 1
            return False
        return True

    def start(self) -> None:
        """Lance le thread d'écriture."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="api-writer", daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Vide la file puis arrête le thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None
        self.flush()

    def flush(self) -> int:
        """Écrit tout ce qui attend (appelé par le thread, ou par les tests) ; renvoie le nombre de lignes écrites."""
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for _ in range(self.batch):
            try:
                table, row = self._queue.get_nowait()
            except queue.Empty:
                break
            grouped[table].append(row)
        written = 0
        for table in ("api_requests", "predictions", "errors"):  # la requête avant ses prédictions
            rows = grouped.get(table)
            if not rows:
                continue
            try:
                self.store.insert(table, rows)
                written += len(rows)
            except SQLAlchemyError as exc:
                with self._lock:
                    self.failed += len(rows)
                logger.warning("%d ligne(s) %s non écrites : %s", len(rows), table, type(exc).__name__)
        with self._lock:
            self.written += written
        if self.pending and written:
            written += self.flush()
        return written

    def _run(self) -> None:
        while not self._stop.wait(self.flush_s):
            self.flush()


def to_json(value: object) -> str:
    """JSON compact (dates en texte, NaN en null)."""
    return json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _version_key(version: str) -> tuple[int, str]:
    return (int(version), version) if version.isdigit() else (0, version)
