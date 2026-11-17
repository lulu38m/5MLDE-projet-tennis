"""Le service de prédiction : assemble historique, modèle et traces ; c'est lui que l'API HTTP appelle.

```
requête ─► profils + overrides ─► FeatureBuilder.features_for ─► pipeline (alias production) ─► réponse
                                                                                  └─► predictions (en arrière-plan)
```

- **Aucune fuite du futur** : on ne prédit qu'un match postérieur à la dernière journée de l'historique ; une date
  plus ancienne est refusée (`422`), elle mélangerait le résultat dans ses propres features.
- **Profil par défaut, overrides possibles** : l'identifiant d'un joueur suffit (dernier classement, âge vieilli,
  taille, main) ; le scoring envoie le classement publié pour le tournoi, qui remplace le dernier connu.
- **Rafraîchissement** : un thread relit l'alias `production` et l'empreinte des matchs à intervalle régulier ;
  `reload()` (`POST /admin/reload`) le fait tout de suite.
"""

from __future__ import annotations

import logging
import math
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from tennis_mlops.config import ApiConfig
from tennis_mlops.features.builder import FEATURE_COLUMNS, PLAYED_STATUSES, MatchContext, PlayerSnapshot
from tennis_mlops.registry.loader import ServedModel
from tennis_mlops.serving.drift import DRIFT_FEATURES, DriftMonitor, DriftReport
from tennis_mlops.serving.features import FeatureSnapshot, FeatureStore
from tennis_mlops.serving.model import ModelHolder
from tennis_mlops.serving.persistence import EPSILON, ApiStore, AsyncWriter, to_json
from tennis_mlops.serving.schemas import (
    FeedbackOutcome,
    FixturePlayer,
    FixturePrediction,
    FixturesResponse,
    H2HResponse,
    MatchRequest,
    ModelRef,
    PlayerInput,
    PlayerOutput,
    PlayerRef,
    PlayersResponse,
    PlayerStats,
    Prediction,
)

logger = logging.getLogger(__name__)
OVERRIDES = ("rank", "rank_points", "age", "height_cm", "hand")


class ServiceUnavailableError(RuntimeError):
    """Le service ne peut pas prédire pour l'instant (pas de modèle, historique pas encore chargé)."""


class InvalidMatchError(ValueError):
    """La requête est bien formée mais ne peut pas être prédite (date passée, batch trop gros…)."""

    def __init__(self, message: str, index: int | None = None) -> None:
        super().__init__(message)
        self.index = index


class UnknownMatchError(LookupError):
    """Résultat d'un match dont le vainqueur n'est aucun des deux joueurs prédits."""


class UnknownPlayerError(LookupError):
    """Joueur sans aucun match joué dans l'historique validé (identifiant faux, ou débutant)."""


@dataclass(frozen=True)
class PredictionBatch:
    """Le résultat d'un appel : la version et l'historique utilisés, et une prédiction par match."""

    served: ServedModel
    snapshot: FeatureSnapshot
    predictions: list[Prediction]

    @property
    def model(self) -> ModelRef:
        """La version, pour la réponse."""
        return ModelRef(name=self.served.name, version=self.served.version, run_id=self.served.run_id)


@dataclass(frozen=True)
class Readiness:
    """Peut-on servir, et dans quel état."""

    status: str  # ready, degraded, not_ready
    checks: dict[str, Any]
    reasons: list[str]

    @property
    def serving(self) -> bool:
        """Vrai si les prédictions répondent (même en mode dégradé)."""
        return self.status != "not_ready"


@dataclass
class PredictionService:
    """Historique, modèle, traces et rafraîchissement périodique."""

    features: FeatureStore
    model: ModelHolder
    store: ApiStore
    writer: AsyncWriter
    config: ApiConfig
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _refresh_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    drift: DriftMonitor = field(init=False)

    def __post_init__(self) -> None:
        self.drift = DriftMonitor(self.config.drift_window, self.config.drift_reference_days)

    # ------------------------------------------------------------------ cycle de vie

    def start(self) -> None:
        """Crée les tables, lance l'écrivain et le thread de rafraîchissement (premier chargement compris)."""
        try:
            self.store.create_schema()
        except Exception as exc:
            logger.error("tables de l'API non créées : %s", exc)
        self.writer.start()
        self._thread = threading.Thread(target=self._loop, name="api-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Arrête le rafraîchissement et vide la file d'écriture."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
        self.writer.stop()

    def reload(self) -> dict[str, Any]:
        """Relit tout de suite l'historique et l'alias (`POST /admin/reload`)."""
        with self._refresh_lock:
            features = self.features.refresh()
            outcome = self.model.refresh()
        model = {"changed": outcome.changed, "version": outcome.version, "previous_version": outcome.previous_version}
        last_day = _iso(features.snapshot.last_day) if features.snapshot else None
        return {
            "model": {**model, "error": outcome.error},
            "features": {"changed": features.changed, "error": features.error, "last_day": last_day},
        }

    def _loop(self) -> None:
        next_model = next_features = 0.0
        while not self._stop.is_set():
            now = _monotonic()
            with self._refresh_lock:
                if now >= next_features:
                    self.features.refresh()
                    next_features = now + (self.config.features_refresh_s if self.features.current else 5)
                if now >= next_model:
                    self.model.refresh()
                    next_model = now + (self.config.model_refresh_s if self.model.served else 5)
            self._stop.wait(1.0)

    # ------------------------------------------------------------------ prédiction

    def predict(self, requests: Sequence[MatchRequest]) -> PredictionBatch:
        """Prédit chaque match ; `ServiceUnavailableError` / `InvalidMatchError` sinon."""
        if len(requests) > self.config.max_batch:
            msg = f"{len(requests)} matchs dans le batch : {self.config.max_batch} au plus"
            raise InvalidMatchError(msg)
        snapshot = self.features.current
        if snapshot is None:
            msg = "historique des matchs pas encore chargé (base injoignable ?)"
            raise ServiceUnavailableError(msg)
        served = self.model.served
        if served is None:
            status = self.model.status()
            reason = status.incompatible or status.error or "aucune version désignée par l'alias production"
            msg = f"aucun modèle servi : {reason}"
            raise ServiceUnavailableError(msg)
        rows, prepared = [], []
        for index, request in enumerate(requests):
            day = request.match_date or snapshot.default_match_date()
            if snapshot.last_day is not None and day <= snapshot.last_day:
                msg = (f"match du {day} : l'historique va jusqu'au {snapshot.last_day}, on ne prédit que des matchs "
                       "postérieurs (sinon le résultat entrerait dans ses propres features)")  # fmt: skip
                raise InvalidMatchError(msg, index)
            warnings: list[str] = []
            p1 = self._player(snapshot, request.player1, day, "player1", warnings)
            p2 = self._player(snapshot, request.player2, day, "player2", warnings)
            context = MatchContext(day, request.surface, request.tourney_level, request.round, request.best_of)
            rows.append(snapshot.builder.features_for(context, p1, p2))
            prepared.append((request, day, p1, p2, warnings))
        frame = pd.concat(rows, ignore_index=True)[list(FEATURE_COLUMNS)]
        probabilities = served.model.predict_proba(frame)[:, 1]
        self.drift.observe(frame[[c for c in DRIFT_FEATURES if c in frame]].to_dict("records"))  # type: ignore[arg-type]
        out = []
        for (request, day, p1, p2, warnings), probability, features in zip(
            prepared, probabilities, frame.to_dict("records"), strict=True
        ):
            p = float(probability)
            out.append(
                Prediction(
                    prediction_id=uuid.uuid4().hex,
                    match_id=request.match_id,
                    match_date=day,
                    player1=self._output(snapshot, p1, p),
                    player2=self._output(snapshot, p2, 1 - p),
                    predicted_winner_id=p1.player_id if p >= 0.5 else p2.player_id,
                    features={str(k): _clean(v) for k, v in features.items()},
                    features_as_of=snapshot.last_day,
                    warnings=warnings,
                )
            )
        return PredictionBatch(served, snapshot, out)

    def record(
        self, batch: PredictionBatch, requests: Sequence[MatchRequest], request_id: str, client: str | None
    ) -> None:
        """Dépose les prédictions dans la file d'écriture."""
        now = datetime.now(UTC)
        data_end = _tag_date(batch.served.tags.get("data_window_end"))
        for request, prediction in zip(requests, batch.predictions, strict=True):
            self.writer.put("predictions", {
                "prediction_id": prediction.prediction_id, "request_id": request_id, "created_at": now,
                "client": client, "match_id": prediction.match_id, "match_date": prediction.match_date,
                "surface": request.surface, "tourney_level": request.tourney_level, "round": request.round,
                "best_of": request.best_of, "tourney_name": request.tourney_name, "p1_id": prediction.player1.id,
                "p2_id": prediction.player2.id,
                "p1_name": prediction.player1.name, "p2_name": prediction.player2.name,
                "p1_win_probability": prediction.player1.win_probability,
                "predicted_winner_id": prediction.predicted_winner_id, "model_name": batch.served.name,
                "model_version": batch.served.version, "run_id": batch.served.run_id,
                "features_as_of": prediction.features_as_of, "batch_size": len(batch.predictions),
                "features": to_json(prediction.features), "warnings": to_json(prediction.warnings),
                "model_data_end": data_end,
            })  # fmt: skip

    @staticmethod
    def _player(
        snapshot: FeatureSnapshot, given: PlayerInput, day: date, label: str, warnings: list[str]
    ) -> PlayerSnapshot:
        profile = snapshot.profiles.get(given.id)
        base = profile.snapshot(day) if profile else PlayerSnapshot(given.id)
        values = {"rank": base.rank, "rank_points": base.rank_points, "age": base.age, "height_cm": base.height_cm,
                  "hand": base.hand}  # fmt: skip
        for key in OVERRIDES:
            value = getattr(given, key)
            if value is not None:
                values[key] = float(value) if key not in {"hand"} else value
        if not snapshot.builder.known(given.id):
            warnings.append(f"{label} ({given.id}) : aucun match dans l'historique, Elo de départ")
        if values["rank"] is None:
            warnings.append(f"{label} ({given.id}) : classement inconnu")
        return PlayerSnapshot(given.id, base.name, values["rank"], values["rank_points"], values["age"],  # type: ignore[arg-type]
                              values["height_cm"], values["hand"])  # type: ignore[arg-type]  # fmt: skip

    @staticmethod
    def _output(snapshot: FeatureSnapshot, player: PlayerSnapshot, probability: float) -> PlayerOutput:
        return PlayerOutput(id=player.player_id, name=player.name, win_probability=round(probability, 6),
                            rank=player.rank, known=snapshot.builder.known(player.player_id))  # fmt: skip

    # ------------------------------------------------------------------ feedback

    def feedback(
        self, match_id: str, winner_id: int, request_id: str, client: str | None
    ) -> tuple[bool, list[FeedbackOutcome]]:
        """Enregistre le vrai résultat et juge les prédictions de ce match."""
        self.writer.flush()  # une prédiction encore en file doit être jugée elle aussi
        known = self.store.predictions_for(match_id)
        players = {p for row in known for p in (row["p1_id"], row["p2_id"])}
        if known and winner_id not in players:
            msg = f"le joueur {winner_id} ne joue pas le match {match_id} (prédit : {sorted(players)})"
            raise UnknownMatchError(msg)
        updated = self.store.record_feedback(match_id, winner_id, client, request_id)
        outcomes = []
        for row in known:
            p1 = float(row["p1_win_probability"])
            winner_probability = p1 if winner_id == row["p1_id"] else 1 - p1
            outcomes.append(
                FeedbackOutcome(
                    prediction_id=row["prediction_id"],
                    model_version=row["model_version"],
                    predicted_winner_id=row["predicted_winner_id"],
                    winner_probability=round(winner_probability, 6),
                    correct=row["predicted_winner_id"] == winner_id,
                    log_loss=round(-math.log(max(winner_probability, EPSILON)), 6),
                )
            )
        return updated, outcomes

    # ------------------------------------------------------------------ consultation (interface de test)

    def _history(self) -> FeatureSnapshot:
        snapshot = self.features.current
        if snapshot is None:
            msg = "historique des matchs pas encore chargé (base injoignable ?)"
            raise ServiceUnavailableError(msg)
        return snapshot

    def search_players(self, query: str | None, limit: int = 20) -> PlayersResponse:
        """`GET /players` : par nom (sans accents ni casse) ou identifiant ; sans texte, les joueurs classés actifs."""
        snapshot = self._history()
        found = snapshot.history.search(snapshot.summaries, query, limit)
        return PlayersResponse(query=query, as_of=snapshot.last_day, count=len(found), players=found)

    def player_stats(self, player_id: int, as_of: date | None = None) -> PlayerStats:
        """`GET /players/{id}/stats` : bilans, forme, classement, derniers résultats (jamais après l'historique)."""
        snapshot = self._history()
        profile = snapshot.summaries.get(player_id)
        if profile is None:
            msg = (f"joueur {player_id} : aucun match joué dans l'historique validé (identifiant inconnu, ou "
                   "débutant : il sera prédit avec l'Elo de départ)")  # fmt: skip
            raise UnknownPlayerError(msg)
        ratings = snapshot.builder.ratings(player_id)
        return snapshot.history.stats(profile, as_of, ratings, form_window=snapshot.builder.config.form_window)

    def head_to_head(self, p1: int, p2: int, as_of: date | None = None) -> H2HResponse:
        """`GET /h2h` : rencontres validées des deux joueurs (un joueur inconnu a simplement zéro rencontre)."""
        if p1 == p2:
            msg = "p1 et p2 doivent être deux joueurs différents"
            raise InvalidMatchError(msg)
        snapshot = self._history()
        refs = [PlayerRef(id=pid, name=getattr(snapshot.summaries.get(pid), "name", None)) for pid in (p1, p2)]
        return snapshot.history.h2h(refs[0], refs[1], as_of)

    def fixture_predictions(self, days: int = 14, client: str | None = "scoring") -> FixturesResponse:
        """`GET /fixtures/predictions` : matchs prédits à l'avance depuis `days` jours, et leur résultat validé."""
        snapshot = self._history()
        anchor = max(d for d in (snapshot.watermark, snapshot.last_day, date.min) if d is not None)
        since = anchor - timedelta(days=days) if anchor != date.min else date.min
        out = [self._fixture(row, snapshot) for row in self.store.fixture_predictions(since, client or None)]
        judged = [f for f in out if f.correct is not None]
        correct = sum(1 for f in judged if f.correct)
        return FixturesResponse(
            as_of=snapshot.last_day, watermark=snapshot.watermark, since=since, client=client or None,
            count=len(out), played=len(judged), correct=correct,
            accuracy=round(correct / len(judged), 4) if judged else None, fixtures=out,
        )  # fmt: skip

    @staticmethod
    def _fixture(row: dict[str, Any], snapshot: FeatureSnapshot) -> FixturePrediction:
        p1 = float(row["p1_win_probability"])
        status, winner, correct, winner_probability = "upcoming", None, None, None
        result = row["result_status"]
        match_day = _as_date(row["match_date"])
        if result in PLAYED_STATUSES:
            status, winner = "played", int(row["actual_winner_id"])
            correct = winner == row["predicted_winner_id"]
            winner_probability = round(p1 if winner == row["p1_id"] else 1 - p1, 6)
        elif result is not None:
            status, winner = "walkover", int(row["actual_winner_id"])
        elif snapshot.last_day is not None and match_day <= snapshot.last_day:
            status = "awaiting_result"
        return FixturePrediction(
            match_id=row["match_id"], match_date=match_day,
            tourney_name=row["tourney_name"] or row["result_tourney_name"], tourney_level=row["tourney_level"],
            surface=row["surface"], round=row["round"],
            player1=FixturePlayer(id=row["p1_id"], name=row["p1_name"], win_probability=round(p1, 6)),
            player2=FixturePlayer(id=row["p2_id"], name=row["p2_name"], win_probability=round(1 - p1, 6)),
            predicted_winner_id=row["predicted_winner_id"], model_version=row["model_version"],
            predicted_at=row["created_at"], client=row["client"], status=status,  # type: ignore[arg-type]
            actual_winner_id=winner, score=row["score"], correct=correct, winner_probability=winner_probability,
        )  # fmt: skip

    # ------------------------------------------------------------------ état

    def readiness(self) -> Readiness:
        """État du service : `ready`, `degraded` (on sert, mais registre ou base injoignables) ou `not_ready`."""
        status = self.model.status()
        snapshot = self.features.current
        reasons: list[str] = []
        if status.served is None:
            reasons.append(status.incompatible or status.error or "aucune version désignée par l'alias production")
        if snapshot is None:
            reasons.append(self.features.last_error or "historique pas encore chargé")
        if reasons:
            state = "not_ready"
        else:
            if status.stale:
                reasons.append(f"registre MLflow injoignable, v{status.served.version} servie depuis le cache : "  # type: ignore[union-attr]
                               f"{status.error}")  # fmt: skip
            if self.features.last_error:
                reasons.append(f"base injoignable, historique du {_iso(snapshot.last_day)} servi : "  # type: ignore[union-attr]
                               f"{self.features.last_error}")  # fmt: skip
            state = "degraded" if reasons else "ready"
        checks = {
            "model": status.served.version if status.served else None,
            "registry": "stale" if status.stale else ("ok" if status.last_success else "unknown"),
            "features_last_day": _iso(snapshot.last_day) if snapshot else None,
            "database": "error" if self.features.last_error else ("ok" if snapshot else "unknown"),
            "writer_pending": self.writer.pending,
        }
        return Readiness(state, checks, reasons)

    def model_info(self) -> dict[str, Any]:
        """`GET /model` : version servie, provenance, état, historique et qualité en ligne."""
        status = self.model.status()
        served = status.served
        snapshot = self.features.current
        info: dict[str, Any] = {"served": None}
        if served is not None:
            tags = {k: v for k, v in served.tags.items() if k in TAGS_SHOWN}
            info["served"] = {
                "name": served.name, "version": served.version, "run_id": served.run_id,
                "alias": self.model.loader.alias.value, "loaded_at": served.loaded_at, "tags": tags,
            }  # fmt: skip
            try:
                online = self.store.online_metrics(served.version)
                info["online"] = online[0].as_dict() if online else None
            except Exception as exc:
                info["online"] = {"error": type(exc).__name__}
        info["registry"] = {
            "last_check": _iso(status.last_check), "last_success": _iso(status.last_success), "stale": status.stale,
            "error": status.error, "incompatible": status.incompatible, "reloads": self.model.reloads,
            "load_seconds": round(status.load_seconds, 2) if status.load_seconds else None,
        }  # fmt: skip
        info["features"] = snapshot.as_dict() if snapshot else None
        info["limits"] = {"max_batch": self.config.max_batch}
        info["drift"] = self.drift_report().as_dict()
        return info

    def drift_report(self) -> DriftReport:
        """Dérive des dernières entrées servies contre la période d'entraînement du modèle servi."""
        snapshot, served = self.features.current, self.model.served
        data_end = _tag_date(served.tags.get("data_window_end")) if served else None
        return self.drift.report(snapshot.history_features if snapshot else None, data_end)


TAGS_SHOWN = ("lifecycle", "candidate", "promoted_at", "decided_by", "gate_status", "features_version",
              "data_window_end", "simulated_now", "dataset_version", "test.log_loss", "git_sha")  # fmt: skip


def _tag_date(value: object) -> date | None:
    """Une date de tag MLflow (`2015-12-31`, ou un horodatage ISO), ou None si absente ou illisible."""
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _clean(value: object) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    if isinstance(value, int | bool):
        return float(value)
    return str(value)


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _iso(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat(timespec="seconds") if isinstance(value, datetime) else value.isoformat()


def _monotonic() -> float:
    import time

    return time.monotonic()
