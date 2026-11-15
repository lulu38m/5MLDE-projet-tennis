"""L'historique des joueurs que l'interface consulte : recherche, statistiques, face-à-face.

Construit à chaque rejeu de l'historique (avec l'instantané des features), à partir des seuls matchs **validés** en
base : on n'y trouve jamais un résultat que l'ingestion n'a pas encore accepté, donc jamais rien d'après l'horloge
simulée. Une date `as_of` plus ancienne restreint encore les statistiques aux matchs joués jusqu'à ce jour.

Un match par joueur et par ligne (format « long ») : un match donne deux lignes, une pour chaque joueur, triées par
joueur puis par date. Les matchs d'un joueur sont une tranche contiguë, trouvée par recherche dichotomique.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from tennis_mlops.features.builder import PLAYED_STATUSES, PlayerRatings
from tennis_mlops.serving.schemas import (
    FormSummary,
    H2HMeeting,
    H2HResponse,
    H2HSurface,
    MatchLine,
    PlayerRef,
    PlayerStats,
    PlayerSummary,
    RankingPoint,
    Ratings,
    Record,
)

LEVELS = {"G": "Grand Chelem", "M": "Masters 1000", "A": "ATP 250 / 500", "F": "Masters de fin d'année",
          "D": "Coupe Davis", "O": "Jeux olympiques"}  # fmt: skip
SURFACES = {"Hard": "Dur", "Clay": "Terre battue", "Grass": "Gazon", "Carpet": "Moquette"}
YEAR_DAYS = 365.25
COLUMNS = ["player_id", "match_date", "tourney_id", "match_num", "match_id", "tourney_name", "tourney_level",
           "surface", "round", "score", "won", "name", "rank", "rank_points", "age", "ht", "hand", "ioc",
           "opponent_id", "opponent_name", "opponent_rank"]  # fmt: skip


class PlayerHistory:
    """Les matchs joués de chaque joueur, prêts à être filtrés par date ou par adversaire."""

    def __init__(self, matches: pd.DataFrame, profiles: Mapping[int, PlayerSummary] | None = None) -> None:
        self.frame = _long(matches)
        self._ids = self.frame["player_id"].to_numpy()
        self._names = {pid: normalize(p.name or "") for pid, p in (profiles or {}).items()}
        self.last_day: date | None = self.frame["match_date"].max() if not self.frame.empty else None

    def __len__(self) -> int:
        return len(self.frame)

    def matches_of(self, player_id: int, as_of: date | None = None) -> pd.DataFrame:
        """Matchs joués du joueur (jusqu'à `as_of` compris), du plus ancien au plus récent."""
        start, end = np.searchsorted(self._ids, player_id, "left"), np.searchsorted(self._ids, player_id, "right")
        part = self.frame.iloc[int(start) : int(end)]
        return part if as_of is None else part[part["match_date"] <= as_of]

    # ------------------------------------------------------------------ recherche

    def search(
        self, profiles: Mapping[int, PlayerSummary], query: str | None, limit: int, active_days: int = 365
    ) -> list[PlayerSummary]:
        """Joueurs dont le nom contient tous les mots de `query` (sans accents ni casse), ou un identifiant exact.

        Sans `query` : les joueurs classés ayant joué dans l'année, par classement (la liste par défaut de l'UI).
        """
        text = (query or "").strip()
        if text.isdigit():
            found = profiles.get(int(text))
            return [found] if found else []
        if text:
            words = normalize(text).split()
            hits = [p for pid, p in profiles.items() if all(w in self._names.get(pid, "") for w in words)]
        else:
            horizon = (self.last_day or date.max) - timedelta(days=active_days)
            hits = [p for p in profiles.values() if p.rank is not None and p.last_match and p.last_match >= horizon]
        hits.sort(key=lambda p: (p.rank is None, p.rank or 0, -p.matches, p.name or ""))
        return hits[:limit]

    # ------------------------------------------------------------------ statistiques d'un joueur

    def stats(
        self,
        profile: PlayerSummary,
        as_of: date | None,
        ratings: PlayerRatings | None,
        form_window: int = 10,
        recent: int = 15,
    ) -> PlayerStats:
        """Bilans, forme, classement dans le temps et derniers résultats, jusqu'à `as_of` (borné à l'historique)."""
        day = min(as_of, self.last_day) if as_of and self.last_day else (self.last_day or as_of or date.today())
        played = self.matches_of(profile.id, day)
        player = _summary_at(profile, played, day)
        year_ago = day - timedelta(days=364)
        last = played.tail(form_window)["won"].astype(bool).tolist()
        shown = None
        if ratings is not None and ratings.matches > 0 and day == self.last_day:
            shown = Ratings(elo=ratings.elo, matches=ratings.matches, surface_elo=ratings.surface_elo,
                            surface_matches=ratings.surface_matches, form=ratings.form)  # fmt: skip
        return PlayerStats(
            player=player,
            as_of=day,
            history_last_day=self.last_day,
            ratings=shown,
            overall=_record(played, "all", "Toutes surfaces"),
            last_52_weeks=_record(played[played["match_date"] > year_ago], "52w", "52 dernières semaines"),
            by_surface=[_record(g, str(k), SURFACES.get(str(k), str(k))) for k, g in _groups(played, "surface")],
            by_level=[_record(g, str(k), LEVELS.get(str(k), str(k))) for k, g in _groups(played, "tourney_level")],
            form=FormSummary(window=form_window, wins=sum(last), losses=len(last) - sum(last), results=last),
            ranking_history=_rankings(played),
            recent=[_line(r) for r in played.tail(recent).iloc[::-1].to_dict("records")],  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ face-à-face

    def h2h(self, p1: PlayerRef, p2: PlayerRef, as_of: date | None = None) -> H2HResponse:
        """Les rencontres des deux joueurs (vues du côté de `p1`), de la plus récente à la plus ancienne."""
        rows = self.matches_of(p1.id, as_of)
        rows = rows[rows["opponent_id"] == p2.id]
        wins = int(rows["won"].sum())
        by_surface = [
            H2HSurface(surface=str(k), p1_wins=int(g["won"].sum()), p2_wins=int((~g["won"]).sum()))
            for k, g in _groups(rows, "surface")
        ]
        meetings = [_meeting(r, p1.id if r["won"] else p2.id) for r in rows.iloc[::-1].to_dict("records")]
        return H2HResponse(player1=p1, player2=p2, as_of=min(as_of, self.last_day) if as_of and self.last_day
                           else self.last_day, total=len(rows), p1_wins=wins, p2_wins=len(rows) - wins,
                           by_surface=by_surface, meetings=meetings)  # fmt: skip


def normalize(text: str) -> str:
    """Minuscules sans accents : « Gaël Monfils » et « gael monfils » se retrouvent."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


# ---------------------------------------------------------------------- calculs


def _long(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=COLUMNS)
    played = matches[matches["status"].isin(PLAYED_STATUSES)]
    common = ["match_date", "tourney_id", "match_num", "match_id", "tourney_name", "tourney_level", "surface",
              "round", "score"]  # fmt: skip
    own = ["id", "name", "rank", "rank_points", "age", "ht", "hand", "ioc"]
    sides = []
    for side, other, won in (("winner", "loser", True), ("loser", "winner", False)):
        part = played[common + [f"{side}_{c}" for c in own] + [f"{other}_id", f"{other}_name", f"{other}_rank"]]
        part = part.set_axis(common + own + ["opponent_id", "opponent_name", "opponent_rank"], axis=1)
        sides.append(part.assign(won=won))
    long = pd.concat(sides, ignore_index=True).rename(columns={"id": "player_id"}).dropna(subset=["player_id"])
    long["player_id"] = long["player_id"].astype("int64")
    long["opponent_id"] = pd.to_numeric(long["opponent_id"], errors="coerce").fillna(0).astype("int64")
    long["match_date"] = pd.to_datetime(long["match_date"]).dt.date
    long["won"] = long["won"].astype(bool)
    return long.sort_values(["player_id", "match_date", "tourney_id", "match_num"], kind="stable")[COLUMNS]\
        .reset_index(drop=True)  # fmt: skip


def _summary_at(profile: PlayerSummary, played: pd.DataFrame, day: date) -> PlayerSummary:
    """Le profil tel qu'il était à `day` : classement de son dernier match jusque-là, âge vieilli jusqu'à `day`."""
    if played.empty:
        return profile.model_copy(update={"rank": None, "rank_points": None, "age": None, "last_match": None,
                                          "matches": 0})  # fmt: skip
    last = played.iloc[-1]
    age = _number(last["age"])
    if age is not None:
        age += (day - last["match_date"]).days / YEAR_DAYS
    return profile.model_copy(update={
        "rank": _number(last["rank"]), "rank_points": _number(last["rank_points"]), "age": _round(age),
        "last_match": last["match_date"], "matches": len(played),
    })  # fmt: skip


def _record(rows: pd.DataFrame, key: str, label: str) -> Record:
    wins = int(rows["won"].sum())
    total = len(rows)
    return Record(key=key, label=label, wins=wins, losses=total - wins,
                  win_rate=round(wins / total, 4) if total else None)  # fmt: skip


def _groups(rows: pd.DataFrame, column: str) -> list[tuple[object, pd.DataFrame]]:
    """Groupes par valeur, du plus fourni au moins fourni (valeurs manquantes ignorées)."""
    groups = [(k, g) for k, g in rows.dropna(subset=[column]).groupby(column, sort=False)]
    return sorted(groups, key=lambda kg: (-len(kg[1]), str(kg[0])))


def _rankings(played: pd.DataFrame) -> list[RankingPoint]:
    """Un point par tournoi : le classement publié à son début (premier match du joueur dans le tournoi)."""
    first = played.drop_duplicates("tourney_id", keep="first")
    first = first[first["rank"].notna()]
    return [RankingPoint(date=r["match_date"], rank=_number(r["rank"]), rank_points=_number(r["rank_points"]))
            for r in first.to_dict("records")]  # fmt: skip


def _meeting(r: Mapping[Any, Any], winner_id: int) -> H2HMeeting:
    return H2HMeeting(
        match_id=str(r["match_id"]), match_date=r["match_date"], tourney_name=_text(r["tourney_name"]),
        tourney_level=_text(r["tourney_level"]), surface=_text(r["surface"]), round=_text(r["round"]),
        winner_id=winner_id, score=_text(r["score"]),
    )  # fmt: skip


def _line(r: Mapping[str, object]) -> MatchLine:
    return MatchLine(
        match_id=str(r["match_id"]), match_date=r["match_date"], tourney_name=_text(r["tourney_name"]),  # type: ignore[arg-type]
        tourney_level=_text(r["tourney_level"]), surface=_text(r["surface"]), round=_text(r["round"]),
        opponent_id=int(r["opponent_id"]), opponent_name=_text(r["opponent_name"]),  # type: ignore[call-overload]
        opponent_rank=_number(r["opponent_rank"]), won=bool(r["won"]), score=_text(r["score"]),
    )  # fmt: skip


def _number(value: object) -> float | None:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _height(value: object) -> float | None:
    height = _number(value)
    return height if height is not None and 150 <= height <= 230 else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _text(value: object) -> str | None:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)) or value == "":
        return None
    return str(value)
