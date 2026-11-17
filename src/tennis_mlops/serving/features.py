"""L'historique rejoué que l'API interroge : état des joueurs (Elo, forme, face-à-face) et dernier profil connu.

Au démarrage, puis à chaque changement des matchs en base (empreinte de la table `matches`), l'historique validé est
rejoué en entier par le même `FeatureBuilder` que l'entraînement (quelques secondes pour 47 000 matchs). Le nouvel
instantané remplace l'ancien d'un coup : une requête voit toujours un état complet et cohérent. Si la base ne répond
pas, l'instantané précédent continue de servir.

Pas de rejeu incrémental : un lot arrive à la fin de son tournoi, avec des matchs plus anciens que ceux d'un tournoi
déjà intégré ; les ajouter à la suite casserait l'ordre chronologique dont dépendent les features.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import pandas as pd

from tennis_mlops.features import FeatureBuilder, FeatureConfig
from tennis_mlops.features.builder import PLAYED_STATUSES, PlayerSnapshot
from tennis_mlops.serving.drift import DRIFT_FEATURES
from tennis_mlops.serving.history import PlayerHistory
from tennis_mlops.serving.schemas import PlayerSummary

logger = logging.getLogger(__name__)

YEAR_DAYS = 365.25


class HistorySource(Protocol):
    """Les matchs validés (PostgreSQL en vrai, un DataFrame dans les tests)."""

    def matches_frame(self) -> pd.DataFrame:
        """Tous les matchs stockés, schéma de la table `matches`."""
        ...

    def lots_summary(self) -> dict[str, Any]:
        """Bilan des lots (dont le watermark : jusqu'où l'ingestion est allée)."""
        ...

    def fingerprint(self) -> str:
        """Empreinte des matchs stockés : change dès qu'un lot est ajouté."""
        ...


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    """Dernier état publié d'un joueur, tiré de son match le plus récent."""

    player_id: int
    name: str | None
    last_match: date
    rank: float | None
    rank_points: float | None
    age: float | None
    height_cm: float | None
    hand: str | None
    ioc: str | None
    matches: int

    def snapshot(self, on: date) -> PlayerSnapshot:
        """Ce que le modèle reçoit pour un match du `on` : l'âge vieillit, le classement est le dernier connu."""
        age = None if self.age is None else self.age + (on - self.last_match).days / YEAR_DAYS
        return PlayerSnapshot(self.player_id, self.name, self.rank, self.rank_points, age, self.height_cm, self.hand)

    def summary(self) -> PlayerSummary:
        """Pour la recherche de joueurs (`GET /players`)."""
        return PlayerSummary(
            id=self.player_id, name=self.name, ioc=self.ioc, hand=self.hand, height_cm=self.height_cm,
            rank=self.rank, rank_points=self.rank_points, age=None if self.age is None else round(self.age, 1),
            last_match=self.last_match, matches=self.matches,
        )  # fmt: skip


@dataclass(frozen=True)
class FeatureSnapshot:
    """Un historique rejoué complet, servi tel quel jusqu'au suivant."""

    builder: FeatureBuilder
    profiles: Mapping[int, PlayerProfile]
    fingerprint: str
    matches: int
    built_at: datetime
    build_seconds: float
    watermark: date | None = None
    history: PlayerHistory = field(default_factory=lambda: PlayerHistory(pd.DataFrame()))
    history_features: pd.DataFrame | None = None  # features des matchs rejoués : la référence de la dérive
    summaries: Mapping[int, PlayerSummary] = field(default_factory=dict)

    @property
    def last_day(self) -> date | None:
        """Dernière journée intégrée : on ne prédit que des matchs postérieurs."""
        return self.builder.last_day

    def default_match_date(self) -> date:
        """Date d'un match dont la requête ne précise pas la date : le lendemain du dernier match, ou le watermark."""
        candidates = [d for d in (self.watermark, self.last_day + timedelta(days=1) if self.last_day else None) if d]
        return max(candidates) if candidates else datetime.now(UTC).date()

    def as_dict(self) -> dict[str, Any]:
        """Pour `GET /model` et les logs."""
        return {
            "last_day": self.last_day.isoformat() if self.last_day else None,
            "watermark": self.watermark.isoformat() if self.watermark else None,
            "matches": self.matches,
            "players": len(self.profiles),
            "fingerprint": self.fingerprint,
            "built_at": self.built_at.isoformat(timespec="seconds"),
            "build_seconds": round(self.build_seconds, 2),
        }


@dataclass(frozen=True)
class FeatureRefresh:
    """Ce qu'a fait un rafraîchissement."""

    changed: bool
    snapshot: FeatureSnapshot | None
    error: str | None = None


@dataclass
class FeatureStore:
    """Garde l'instantané servi et le remplace quand les matchs changent."""

    source: HistorySource
    config: FeatureConfig
    _current: FeatureSnapshot | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    last_error: str | None = field(default=None, init=False)
    last_check: datetime | None = field(default=None, init=False)

    @property
    def current(self) -> FeatureSnapshot | None:
        """L'instantané servi (None tant que le premier rejeu n'a pas réussi)."""
        with self._lock:
            return self._current

    def refresh(self, force: bool = False) -> FeatureRefresh:
        """Rejoue l'historique si les matchs ont changé (ou si `force`) ; ne lève jamais."""
        current = self.current
        self.last_check = datetime.now(UTC)
        try:
            fingerprint = self.source.fingerprint()
            if current is not None and not force and current.fingerprint == fingerprint:
                self.last_error = None
                return FeatureRefresh(False, current)
            snapshot = self._build(fingerprint)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__} : {exc}"[:300]
            logger.warning("historique non rafraîchi, l'état précédent reste servi : %s", self.last_error)
            return FeatureRefresh(False, current, self.last_error)
        with self._lock:
            self._current = snapshot
        self.last_error = None
        logger.info("historique rejoué : %s", snapshot.as_dict())
        return FeatureRefresh(True, snapshot)

    def _build(self, fingerprint: str) -> FeatureSnapshot:
        started = time.perf_counter()
        matches = self.source.matches_frame()
        summary = self.source.lots_summary()
        builder = FeatureBuilder(self.config)
        history_features = None
        if not matches.empty:
            built = builder.build(matches)
            history_features = built[["match_date", *[c for c in DRIFT_FEATURES if c in built]]].copy()
        watermark = summary.get("watermark")
        profiles = player_profiles(matches)
        summaries = {pid: p.summary() for pid, p in profiles.items()}
        return FeatureSnapshot(
            builder=builder,
            profiles=profiles,
            fingerprint=fingerprint,
            matches=len(matches),
            built_at=datetime.now(UTC),
            build_seconds=time.perf_counter() - started,
            watermark=date.fromisoformat(str(watermark)[:10]) if watermark else None,
            history=PlayerHistory(matches, summaries),
            summaries=summaries,
            history_features=history_features,
        )


def player_profiles(matches: pd.DataFrame) -> dict[int, PlayerProfile]:
    """Dernier profil de chaque joueur (match joué le plus récent), et son nombre de matchs joués."""
    if matches.empty:
        return {}
    played = matches[matches["status"].isin(PLAYED_STATUSES)]
    columns = ["id", "name", "rank", "rank_points", "age", "ht", "hand", "ioc"]
    sides = []
    for side in ("winner", "loser"):
        part = played[["match_date", "tourney_id", "match_num", *[f"{side}_{c}" for c in columns]]].copy()
        part.columns = ["match_date", "tourney_id", "match_num", *columns]
        sides.append(part)
    long = pd.concat(sides, ignore_index=True).dropna(subset=["id"])
    long["match_date"] = pd.to_datetime(long["match_date"])
    long = long.sort_values(["match_date", "tourney_id", "match_num"], kind="stable")
    counts = long.groupby("id").size()
    last = long.groupby("id").tail(1)
    profiles: dict[int, PlayerProfile] = {}
    for row in last.to_dict("records"):
        player_id = int(row["id"])
        height = _number(row["ht"])
        profiles[player_id] = PlayerProfile(
            player_id=player_id,
            name=_text(row["name"]),
            last_match=pd.Timestamp(row["match_date"]).date(),
            rank=_number(row["rank"]),
            rank_points=_number(row["rank_points"]),
            age=_number(row["age"]),
            height_cm=height if height is not None and 150 <= height <= 230 else None,
            hand=_text(row["hand"]),
            ioc=_text(row["ioc"]),
            matches=int(counts[row["id"]]),
        )
    return profiles


def _number(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return None
    return float(value)  # type: ignore[arg-type]


def _text(value: object) -> str | None:
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    return str(value)
