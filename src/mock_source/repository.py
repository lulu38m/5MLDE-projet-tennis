"""Données rejouées : chargement, préparation et requêtes **toujours filtrées par la date simulée**.

Règles de visibilité (aucune fuite du futur) :

- un **résultat** est visible si `match_date < now` ;
- un **match à venir** est visible si ses deux joueurs sont connus (`known_date <= now`) et qu'il n'est pas joué
  (`match_date >= now`). Un joueur est connu au tour N quand son match du tour N-1 est terminé : on ne révèle jamais
  un quart de finale avant la fin des huitièmes ;
- un **classement** est visible si `ranking_date <= now` ;
- un **tournoi** apparaît dans le calendrier `tournament_notice_days` jours avant son début.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Hashable, Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

from mock_source.schedule import ROUND_ROBIN, estimate_day_offsets
from mock_source.schemas import (
    Fixture,
    MatchResult,
    PlayerEntry,
    PlayerHistory,
    PlayerMatch,
    PlayerProfile,
    RankingEntry,
    RankingPoint,
    TournamentStatus,
    TournamentSummary,
)
from tennis_mlops.config import MockSourceConfig
from tennis_mlops.data.catalog import DatasetCatalog, FileKind
from tennis_mlops.data.errors import DataAccessError
from tennis_mlops.data.lockfile import DataLock
from tennis_mlops.data.score import MatchStatus, classify_score

logger = logging.getLogger(__name__)

PLAYER_FIELDS = ("id", "name", "hand", "ht", "ioc", "age", "rank", "rank_points", "seed", "entry")
MATCH_COLUMNS = (
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "score",
    "best_of",
    "round",
    "minutes",
    *(f"{side}_{field}" for side in ("winner", "loser") for field in PLAYER_FIELDS),
)
_INT_FIELDS = {"id", "ht", "rank", "rank_points", "seed"}


def _clean(value: object) -> object:
    """Convertit NaN/NA pandas en None et les scalaires numpy en types Python."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):  # type: ignore[call-overload]
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [{str(k): _clean(v) for k, v in row.items()} for row in frame.to_dict("records")]


def _to_date(value: object) -> date:
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def _ts(day: date) -> pd.Timestamp:
    return pd.Timestamp(day)


def player1_is_winner(match_id: str) -> bool:
    """Ordre neutre et déterministe des joueurs d'un match à venir (parité du hash du match_id)."""
    return int(hashlib.sha256(match_id.encode("utf-8")).hexdigest()[:8], 16) % 2 == 0


def _player(row: Mapping[str, object], side: str) -> PlayerEntry:
    fields: dict[str, object] = {}
    for field in PLAYER_FIELDS:
        value = row[f"{side}_{field}"]
        if value is not None and field in _INT_FIELDS:
            value = int(value)  # type: ignore[call-overload]
        if value is not None and field == "age":
            value = round(float(value), 1)  # type: ignore[arg-type]
        fields[field] = value
    return PlayerEntry.model_validate(fields)


def _match_info(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "match_id": row["match_id"],
        "tourney_id": row["tourney_id"],
        "tourney_name": row["tourney_name"],
        "tourney_level": row["tourney_level"],
        "surface": row["surface"],
        "draw_size": int(row["draw_size"]),  # type: ignore[call-overload]
        "best_of": int(row["best_of"]),  # type: ignore[call-overload]
        "round": row["round"],
        "match_num": int(row["match_num"]),  # type: ignore[call-overload]
        "match_date": _to_date(row["match_date"]),
    }


def prepare_matches(raw: pd.DataFrame, first_round_notice_days: int) -> pd.DataFrame:
    """Ajoute `match_id`, `match_date`, `known_date`, `status` et l'ordre neutre des joueurs."""
    matches = raw.loc[:, list(MATCH_COLUMNS)].copy()
    matches["match_id"] = matches["tourney_id"].astype(str) + "-" + matches["match_num"].astype(int).astype(str)
    matches["start_date"] = pd.to_datetime(matches["tourney_date"].astype(int).astype(str), format="%Y%m%d")

    offsets = np.zeros(len(matches), dtype="int64")
    positions = matches.groupby("tourney_id", sort=False).indices
    for group_positions in positions.values():
        group = matches.iloc[group_positions]
        offsets[group_positions] = estimate_day_offsets(
            group["round"].tolist(),
            group["match_num"].astype(int).tolist(),
            str(group["tourney_level"].iloc[0]),
            int(group["draw_size"].iloc[0]),
        )
    matches["match_date"] = matches["start_date"] + pd.to_timedelta(offsets, unit="D")
    matches["known_date"] = _known_dates(matches, first_round_notice_days)
    matches["status"] = [classify_score(None if pd.isna(s) else str(s)).value for s in matches["score"]]
    matches["p1_is_winner"] = [player1_is_winner(m) for m in matches["match_id"]]
    return matches.sort_values(["match_date", "match_id"], ignore_index=True)


def _known_dates(matches: pd.DataFrame, notice_days: int) -> pd.Series[pd.Timestamp]:
    """Date à laquelle les deux joueurs d'un match sont connus (fin de leur match précédent dans le tournoi)."""
    known: dict[int, pd.Timestamp] = {}
    notice = pd.Timedelta(days=notice_days)
    one_day = pd.Timedelta(days=1)
    ordered = matches.sort_values(["tourney_id", "match_date", "match_num"])
    last_played: dict[tuple[str, int], pd.Timestamp] = {}
    for index, tourney_id, rnd, winner, loser, start, played in zip(
        ordered.index,
        ordered["tourney_id"],
        ordered["round"],
        ordered["winner_id"],
        ordered["loser_id"],
        ordered["start_date"],
        ordered["match_date"],
        strict=True,
    ):
        previous = [last_played[key] for key in ((tourney_id, winner), (tourney_id, loser)) if key in last_played]
        if rnd == ROUND_ROBIN or not previous:
            known[index] = start - notice
        else:
            known[index] = min(max(previous) + one_day, played)
        last_played[(tourney_id, winner)] = played
        last_played[(tourney_id, loser)] = played
    return pd.Series(known, dtype="datetime64[ns]").reindex(matches.index)


class ReplayRepository:
    """Point d'accès unique aux données rejouées."""

    def __init__(
        self,
        matches: pd.DataFrame,
        rankings: pd.DataFrame,
        players: pd.DataFrame,
        dataset_version: str,
        config: MockSourceConfig,
    ) -> None:
        self._matches = matches
        self._config = config
        self.dataset_version = dataset_version
        self._tournaments = self._build_tournaments(matches)
        self._rankings = rankings.sort_values(["ranking_date", "rank"], ignore_index=True)
        self._ranking_dates = np.sort(self._rankings["ranking_date"].unique())
        self._ranking_by_date = self._rankings.groupby("ranking_date").indices
        self._ranking_by_player = self._rankings.groupby("player").indices
        self._players = players.set_index("player_id", drop=False)
        self._appearances = self._build_appearances(matches)
        self._appearances_by_player = self._appearances.groupby("player_id").indices
        self._names = self._build_names()

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_raw(cls, raw_dir: Path, catalog: DatasetCatalog, lock: DataLock, config: MockSourceConfig) -> Self:
        """Vérifie les fichiers contre le verrou puis les charge (refuse de démarrer sur des données altérées)."""
        check_readable(raw_dir, catalog.names())
        lock.verify_directory(raw_dir, catalog.names())
        raw_matches = pd.concat(
            [
                pd.read_csv(raw_dir / name, dtype={"tourney_id": "string", "score": "string"}, low_memory=False)
                for name in catalog.match_files().values()
            ],
            ignore_index=True,
        )
        rankings = pd.concat(
            [pd.read_csv(raw_dir / f.name) for f in catalog.of_kind(FileKind.RANKINGS)], ignore_index=True
        ).drop_duplicates(["ranking_date", "player"])
        rankings["ranking_date"] = pd.to_datetime(rankings["ranking_date"].astype(str), format="%Y%m%d")
        players = pd.read_csv(raw_dir / DatasetCatalog.PLAYERS_FILE, dtype={"dob": "string"}, low_memory=False)
        matches = prepare_matches(raw_matches, config.first_round_notice_days)
        logger.info("%d matchs, %d lignes de classement chargés", len(matches), len(rankings))
        return cls(matches, rankings, players, lock.dataset_version, config)

    @staticmethod
    def _build_tournaments(matches: pd.DataFrame) -> pd.DataFrame:
        grouped = matches.groupby("tourney_id", sort=False)
        tournaments = grouped.agg(
            tourney_name=("tourney_name", "first"),
            tourney_level=("tourney_level", "first"),
            surface=("surface", "first"),
            draw_size=("draw_size", "first"),
            start_date=("start_date", "first"),
            end_date=("match_date", "max"),
            expected_matches=("match_id", "size"),
        )
        return tournaments.sort_values(["start_date", "tourney_id"])

    @staticmethod
    def _build_appearances(matches: pd.DataFrame) -> pd.DataFrame:
        common = ["match_id", "match_date", "tourney_id", "tourney_name", "tourney_level", "surface", "round", "score"]
        common += ["status"]
        views = []
        for side, other, won in (("winner", "loser", True), ("loser", "winner", False)):
            view = matches[common].copy()
            view["player_id"] = matches[f"{side}_id"].astype(int)
            view["opponent_id"] = matches[f"{other}_id"].astype(int)
            view["opponent_name"] = matches[f"{other}_name"]
            view["won"] = won
            views.append(view)
        return pd.concat(views, ignore_index=True).sort_values(["match_date", "match_id"], ignore_index=True)

    def _build_names(self) -> dict[int, str]:
        names: dict[int, str] = {}
        for side in ("winner", "loser"):
            names.update(
                zip(self._matches[f"{side}_id"].astype(int), self._matches[f"{side}_name"].astype(str), strict=True)
            )
        for player_id, first, last in zip(
            self._players["player_id"], self._players["name_first"], self._players["name_last"], strict=True
        ):
            full = " ".join(str(p) for p in (first, last) if isinstance(p, str))
            names.setdefault(int(player_id), full)
        return names

    # ------------------------------------------------------------------ propriétés

    @property
    def match_count(self) -> int:
        """Nombre de matchs chargés."""
        return len(self._matches)

    @property
    def first_date(self) -> date:
        """Premier jour de match."""
        return _to_date(self._matches["match_date"].min())

    @property
    def last_date(self) -> date:
        """Dernier jour de match."""
        return _to_date(self._matches["match_date"].max())

    # ------------------------------------------------------------------ matchs

    def results(self, since: date, until: date) -> list[MatchResult]:
        """Matchs terminés avec `since <= match_date < until` (until ne doit pas dépasser now)."""
        frame = self._matches[(self._matches["match_date"] >= _ts(since)) & (self._matches["match_date"] < _ts(until))]
        return [self._result(row) for row in _records(frame)]

    def fixtures(self, now: date, horizon_days: int) -> list[Fixture]:
        """Matchs dont les deux joueurs sont connus et qui se jouent dans `[now ; now + horizon[`."""
        m = self._matches
        frame = m[
            (m["known_date"] <= _ts(now))
            & (m["match_date"] >= _ts(now))
            & (m["match_date"] < _ts(now + timedelta(days=horizon_days)))
        ]
        return [self._fixture(row) for row in _records(frame)]

    def count_completed_between(self, start: date, end: date) -> int:
        """Nombre de matchs passés à « terminé » quand l'horloge va de start à end."""
        dates = self._matches["match_date"]
        return int(((dates >= _ts(start)) & (dates < _ts(end))).sum())

    @staticmethod
    def _result(row: Mapping[str, object]) -> MatchResult:
        minutes = row["minutes"]
        return MatchResult(
            **_match_info(row),  # type: ignore[arg-type]
            winner=_player(row, "winner"),
            loser=_player(row, "loser"),
            score=row["score"],  # type: ignore[arg-type]
            minutes=None if minutes is None else int(minutes),  # type: ignore[call-overload]
            status=MatchStatus(str(row["status"])),
        )

    @staticmethod
    def _fixture(row: Mapping[str, object]) -> Fixture:
        winner, loser = _player(row, "winner"), _player(row, "loser")
        first, second = (winner, loser) if row["p1_is_winner"] else (loser, winner)
        return Fixture(**_match_info(row), player1=first, player2=second)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ tournois

    def tournaments(
        self,
        now: date,
        since: date | None = None,
        until: date | None = None,
        status: TournamentStatus | None = None,
    ) -> list[TournamentSummary]:
        """Tournois visibles à `now`, filtrés sur leur date de fin et leur statut."""
        visible = self._visible_tournaments(now)
        if since is not None:
            visible = visible[visible["end_date"] >= _ts(since)]
        if until is not None:
            visible = visible[visible["end_date"] < _ts(until)]
        completed = self._completed_counts(now)
        summaries = [self._summary(str(tid), row, now, completed.get(tid, 0)) for tid, row in visible.iterrows()]
        return [s for s in summaries if status is None or s.status is status]

    def tournament(self, tourney_id: str, now: date) -> tuple[TournamentSummary, list[MatchResult]] | None:
        """Résumé et résultats déjà connus d'un tournoi ; None s'il est inconnu ou pas encore annoncé."""
        visible = self._visible_tournaments(now)
        if tourney_id not in visible.index:
            return None
        m = self._matches
        frame = m[(m["tourney_id"] == tourney_id) & (m["match_date"] < _ts(now))]
        summary = self._summary(tourney_id, visible.loc[[tourney_id]].iloc[0], now, len(frame))
        return summary, [self._result(row) for row in _records(frame)]

    def tournaments_completed_between(self, start: date, end: date) -> list[str]:
        """Tournois dont le dernier match a eu lieu dans `[start ; end[`."""
        t = self._tournaments
        return [str(i) for i in t[(t["end_date"] >= _ts(start)) & (t["end_date"] < _ts(end))].index]

    def _visible_tournaments(self, now: date) -> pd.DataFrame:
        notice = pd.Timedelta(days=self._config.tournament_notice_days)
        return self._tournaments[self._tournaments["start_date"] - notice <= _ts(now)]

    def _completed_counts(self, now: date) -> dict[Hashable, int]:
        played = self._matches[self._matches["match_date"] < _ts(now)]
        return {k: int(v) for k, v in played.groupby("tourney_id").size().items()}

    @staticmethod
    def _summary(tourney_id: str, row: pd.Series[Any], now: date, completed: int) -> TournamentSummary:
        end = _to_date(row["end_date"])
        start = _to_date(row["start_date"])
        if end < now:
            status = TournamentStatus.COMPLETED
        elif completed > 0 or start <= now:
            status = TournamentStatus.IN_PROGRESS
        else:
            status = TournamentStatus.UPCOMING
        surface = row["surface"]
        return TournamentSummary(
            tourney_id=tourney_id,
            tourney_name=str(row["tourney_name"]),
            tourney_level=str(row["tourney_level"]),
            surface=None if pd.isna(surface) else str(surface),
            draw_size=int(row["draw_size"]),
            start_date=start,
            end_date=end,
            status=status,
            expected_matches=int(row["expected_matches"]),
            completed_matches=completed,
        )

    def empty_period_reason(self, since: date, until: date) -> str:
        """Explique une période sans match : pause connue (ex. 2020) ou intersaison."""
        for pause in self._config.pauses:
            if since <= pause.end and until > pause.start:
                return pause.reason
        return "Aucun match programmé sur la période (intersaison ou semaine sans tournoi)"

    # ------------------------------------------------------------------ classements

    def latest_ranking_date(self, at: date) -> date | None:
        """Dernier classement publié à la date `at` (incluse)."""
        position = int(np.searchsorted(self._ranking_dates, np.datetime64(at), side="right")) - 1
        return None if position < 0 else _to_date(self._ranking_dates[position])

    def rankings(self, at: date, limit: int, player_id: int | None = None) -> tuple[date | None, list[RankingEntry]]:
        """Classement publié à la date `at`, limité aux `limit` premiers (ou à un joueur)."""
        ranking_date = self.latest_ranking_date(at)
        if ranking_date is None:
            return None, []
        frame = self._rankings.iloc[self._ranking_by_date[_ts(ranking_date)]]
        if player_id is not None:
            frame = frame[frame["player"] == player_id]
        frame = frame.head(limit)
        return ranking_date, [
            RankingEntry(
                rank=int(row["rank"]),
                player_id=int(row["player"]),
                name=self._names.get(int(row["player"])),
                points=None if pd.isna(row["points"]) else int(row["points"]),
            )
            for _, row in frame.iterrows()
        ]

    # ------------------------------------------------------------------ joueurs

    def player_profile(self, player_id: int) -> PlayerProfile | None:
        """Fiche d'un joueur ; None si inconnu."""
        if player_id not in self._players.index:
            name = self._names.get(player_id)
            return None if name is None else PlayerProfile(id=player_id, name=name)
        row = self._players.loc[player_id]
        dob = row.get("dob")
        height = row.get("height")
        return PlayerProfile(
            id=player_id,
            name=self._names.get(player_id, str(player_id)),
            hand=_optional_str(row.get("hand")),
            dob=_parse_dob(dob),
            ioc=_optional_str(row.get("ioc")),
            height=None if height is None or pd.isna(height) else int(height),
        )

    def player_history(self, player_id: int, as_of: date, match_limit: int, ranking_weeks: int) -> PlayerHistory | None:
        """Historique d'un joueur strictement avant `as_of` (matchs) / jusqu'à `as_of` (classements)."""
        profile = self.player_profile(player_id)
        if profile is None:
            return None
        appearances = self._rows(self._appearances, self._appearances_by_player, player_id)
        played = appearances[appearances["match_date"] < _ts(as_of)]
        rankings = self._rows(self._rankings, self._ranking_by_player, player_id)
        lower = _ts(as_of - timedelta(weeks=ranking_weeks))
        rankings = rankings[(rankings["ranking_date"] <= _ts(as_of)) & (rankings["ranking_date"] > lower)]
        wins = int(played["won"].sum())
        return PlayerHistory(
            player=profile,
            as_of=as_of,
            is_debutant=played.empty,
            wins=wins,
            losses=len(played) - wins,
            rankings=[
                RankingPoint(
                    ranking_date=_to_date(r["ranking_date"]),
                    rank=int(r["rank"]),
                    points=None if pd.isna(r["points"]) else int(r["points"]),
                )
                for _, r in rankings.iterrows()
            ],
            matches=[
                PlayerMatch(
                    match_id=str(r["match_id"]),
                    match_date=_to_date(r["match_date"]),
                    tourney_id=str(r["tourney_id"]),
                    tourney_name=str(r["tourney_name"]),
                    tourney_level=str(r["tourney_level"]),
                    surface=_optional_str(r["surface"]),
                    round=str(r["round"]),
                    opponent_id=int(r["opponent_id"]),
                    opponent_name=str(r["opponent_name"]),
                    won=bool(r["won"]),
                    score=_optional_str(r["score"]),
                    status=MatchStatus(str(r["status"])),
                )
                for _, r in played.tail(match_limit).iloc[::-1].iterrows()
            ],
        )

    @staticmethod
    def _rows(frame: pd.DataFrame, index: Mapping[Hashable, Iterable[int]], key: int) -> pd.DataFrame:
        positions = index.get(key)
        return frame.iloc[list(positions)] if positions is not None else frame.iloc[0:0]


def _parse_dob(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    parsed = pd.to_datetime(value, format="%Y%m%d", errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def check_readable(raw_dir: Path, names: Sequence[str]) -> None:
    """Échoue avec un message actionnable si des fichiers de données ne sont pas lisibles par ce processus."""
    unreadable = [n for n in names if (raw_dir / n).exists() and not os.access(raw_dir / n, os.R_OK)]
    if unreadable:
        uid = getattr(os, "getuid", lambda: "?")()  # pas d'uid sous Windows
        msg = (
            f"{len(unreadable)} fichier(s) de {raw_dir} illisible(s) par l'utilisateur uid={uid} "
            f"(ex. {unreadable[0]}). Sur la machine hôte : `make data` (rétablit les droits) "
            "ou `chmod a+r data/raw/*`, puis relancer."
        )
        raise DataAccessError(msg)


def _optional_str(value: object) -> str | None:
    return None if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)  # type: ignore[call-overload]
