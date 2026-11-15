"""Construction des features d'un match, **strictement avant** sa date. Partagé par l'entraînement et l'API.

Le builder rejoue l'historique dans l'ordre chronologique, une journée à la fois :

1. pour chaque match du jour, il calcule les features avec l'état des joueurs **de la veille** ;
2. puis seulement il met à jour cet état (Elo, forme, face-à-face) avec les résultats du jour.

Deux matchs du même jour ne se voient donc jamais, et le résultat d'un match n'entre jamais dans ses propres
features. Une seule fonction, `_features()`, calcule les features : l'entraînement (`build`) et l'API
(`features_for`) passent par elle, ce qui écarte tout écart entre entraînement et service. Elle ne modifie pas l'état :
l'API peut la lancer depuis plusieurs threads à la fois sur le même builder.

Orientation : la colonne `label` vaut 1 si le **joueur 1** gagne. Qui est joueur 1 est tiré de façon déterministe à
partir du `match_id` (empreinte SHA-256) : environ un match sur deux, le vainqueur est le joueur 2. Sans ça, le
label vaudrait toujours 1 et le modèle n'apprendrait rien.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

FEATURES_VERSION = "1"

#: Statuts qui comptent comme un vrai match joué (état des joueurs + exemples d'entraînement). Un forfait (`walkover`)
#: n'a pas été joué ; un match `defaulted` ou `abandoned` ne dit rien du niveau des joueurs.
PLAYED_STATUSES = frozenset({"completed", "retired"})

NUMERIC_FEATURES: tuple[str, ...] = (
    "rank_log_ratio",
    "points_log_ratio",
    "age_diff",
    "height_diff",
    "elo_diff",
    "surface_elo_diff",
    "elo_win_prob",
    "experience_diff",
    "form_diff",
    "activity_diff",
    "rest_days_diff",
    "h2h_diff",
    "h2h_total",
    "best_of",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("surface", "tourney_level", "round", "p1_hand", "p2_hand")
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Colonnes descriptives gardées à côté des features (segments d'évaluation, analyse d'erreurs) ; jamais en entrée.
META_COLUMNS: tuple[str, ...] = (
    "match_id",
    "match_date",
    "tourney_id",
    "tourney_name",
    "p1_id",
    "p1_name",
    "p1_rank",
    "p2_id",
    "p2_name",
    "p2_rank",
    "status",
)
LABEL = "label"

#: Tailles hors de cette plage traitées comme inconnues (la source contient par exemple « 3 cm »).
HEIGHT_RANGE_CM = (150.0, 230.0)


class FeatureConfig(BaseModel):
    """Paramètres des features (tracés dans MLflow avec `FEATURES_VERSION`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    elo_start: float = Field(default=1500.0, description="Elo d'un joueur jamais vu")
    elo_k_numerator: float = Field(default=250.0, gt=0, description="K = numérateur / (matchs + décalage) ** forme")
    elo_k_offset: float = Field(default=5.0, gt=0)
    elo_k_shape: float = Field(default=0.4, gt=0)
    surface_weight: float = Field(default=0.5, ge=0, le=1, description="Poids de l'Elo surface dans la proba Elo")
    form_window: int = Field(default=10, ge=1, description="Derniers matchs pris en compte pour la forme")
    activity_days: int = Field(default=365, ge=1, description="Fenêtre de comptage des matchs joués")
    max_rest_days: int = Field(default=365, ge=1, description="Plafond du nombre de jours sans jouer")


@dataclass(frozen=True, slots=True)
class PlayerSnapshot:
    """Ce que l'on sait d'un joueur au début du tournoi (classement publié, âge, taille, main)."""

    player_id: int
    name: str | None = None
    rank: float | None = None
    rank_points: float | None = None
    age: float | None = None
    height_cm: float | None = None
    hand: str | None = None


@dataclass(frozen=True, slots=True)
class MatchContext:
    """Le cadre du match, connu avant qu'il se joue."""

    match_date: date
    surface: str | None = None
    tourney_level: str | None = None
    round: str | None = None
    best_of: int | None = None


@dataclass(frozen=True, slots=True)
class PlayerRatings:
    """L'état d'un joueur dans le builder, exposé à l'interface (Elo, forme lissée)."""

    elo: float
    matches: int
    surface_elo: dict[str, float]
    surface_matches: dict[str, int]
    form: float
    last_match: date | None


@dataclass(slots=True)
class _PlayerState:
    elo: float
    matches: int = 0
    surface_elo: dict[str, float] = field(default_factory=dict)
    surface_matches: dict[str, int] = field(default_factory=dict)
    recent: deque[bool] = field(default_factory=deque)
    dates: deque[date] = field(default_factory=deque)
    last_date: date | None = None


def player1_is_winner(match_id: str) -> bool:
    """Orientation déterministe : le vainqueur est joueur 1 pour environ un match sur deux."""
    return int(hashlib.sha256(match_id.encode()).hexdigest()[:8], 16) % 2 == 0


class FeatureBuilder:
    """Rejoue l'historique et calcule les features de chaque match avec l'état de la veille."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._players: dict[int, _PlayerState] = {}
        self._h2h: defaultdict[tuple[int, int], int] = defaultdict(int)  # (vainqueur, perdant) → victoires
        self._last_day: date | None = None

    # ------------------------------------------------------------------ entraînement

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Une ligne par match joué : features, colonnes descriptives et label, dans l'ordre chronologique.

        `matches` suit le schéma de la table `matches` (colonnes `winner_*` / `loser_*`). L'état du builder est
        conservé : un appel suivant (ou `features_for`) part de là où celui-ci s'est arrêté.
        """
        played = matches[matches["status"].isin(PLAYED_STATUSES)]
        ordered = played.sort_values(["match_date", "tourney_id", "match_num"], kind="stable")
        rows: list[dict[str, Any]] = []
        day_results: list[tuple[MatchContext, int, int]] = []
        current_day: date | None = None
        records: list[dict[str, Any]] = ordered.to_dict("records")  # type: ignore[assignment]
        for record in records:
            day = _as_date(record["match_date"])
            if self._last_day is not None and day <= self._last_day:
                msg = f"match du {day} déjà couvert par l'historique (jusqu'au {self._last_day}) : ordre rompu"
                raise ValueError(msg)
            if day != current_day:
                self._apply(day_results)  # les résultats de la veille entrent dans l'état, pas ceux du jour
                day_results, current_day = [], day
            context = MatchContext(
                match_date=day,
                surface=_text(record.get("surface")),
                tourney_level=_text(record.get("tourney_level")),
                round=_text(record.get("round")),
                best_of=_int(record.get("best_of")),
            )
            winner, loser = _snapshot(record, "winner"), _snapshot(record, "loser")
            winner_first = player1_is_winner(str(record["match_id"]))
            p1, p2 = (winner, loser) if winner_first else (loser, winner)
            row = self._features(context, p1, p2)
            row.update(_meta(record, context, p1, p2))
            row[LABEL] = int(winner_first)
            rows.append(row)
            day_results.append((context, winner.player_id, loser.player_id))
        self._apply(day_results)
        frame = pd.DataFrame(rows, columns=[*FEATURE_COLUMNS, *META_COLUMNS, LABEL])
        return _typed(frame)

    # ------------------------------------------------------------------ service (API)

    def features_for(self, context: MatchContext, p1: PlayerSnapshot, p2: PlayerSnapshot) -> pd.DataFrame:
        """Features d'un match à venir, avec l'état actuel du builder (une ligne, colonnes du modèle)."""
        if self._last_day is not None and context.match_date <= self._last_day:
            msg = f"le match du {context.match_date} n'est pas postérieur à l'historique ({self._last_day})"
            raise ValueError(msg)
        frame = pd.DataFrame([self._features(context, p1, p2)], columns=list(FEATURE_COLUMNS))
        return _typed(frame)

    @property
    def last_day(self) -> date | None:
        """Dernière journée intégrée à l'état des joueurs."""
        return self._last_day

    @property
    def players(self) -> int:
        """Nombre de joueurs connus."""
        return len(self._players)

    # ------------------------------------------------------------------ calcul

    def _features(self, context: MatchContext, p1: PlayerSnapshot, p2: PlayerSnapshot) -> dict[str, Any]:
        cfg = self.config
        s1, s2 = self._peek(p1.player_id), self._peek(p2.player_id)  # lecture seule : l'API partage l'état
        surface = context.surface
        e1, e2 = s1.elo, s2.elo
        se1 = s1.surface_elo.get(surface, cfg.elo_start) if surface else e1
        se2 = s2.surface_elo.get(surface, cfg.elo_start) if surface else e2
        blended = (1 - cfg.surface_weight) * (e1 - e2) + cfg.surface_weight * (se1 - se2)
        h2h_won = self._h2h.get((p1.player_id, p2.player_id), 0)
        h2h_lost = self._h2h.get((p2.player_id, p1.player_id), 0)
        return {
            "rank_log_ratio": _log_ratio(p2.rank, p1.rank),
            "points_log_ratio": _log1p_diff(p1.rank_points, p2.rank_points),
            "age_diff": _diff(p1.age, p2.age),
            "height_diff": _diff(p1.height_cm, p2.height_cm),
            "elo_diff": e1 - e2,
            "surface_elo_diff": se1 - se2,
            "elo_win_prob": _expected(blended),
            "experience_diff": math.log1p(s1.matches) - math.log1p(s2.matches),
            "form_diff": self._form(s1) - self._form(s2),
            "activity_diff": float(self._activity(s1, context.match_date) - self._activity(s2, context.match_date)),
            "rest_days_diff": float(self._rest(s1, context.match_date) - self._rest(s2, context.match_date)),
            "h2h_diff": float(h2h_won - h2h_lost),
            "h2h_total": float(h2h_won + h2h_lost),
            "best_of": float(context.best_of) if context.best_of else math.nan,
            "surface": surface,
            "tourney_level": context.tourney_level,
            "round": context.round,
            "p1_hand": p1.hand,
            "p2_hand": p2.hand,
        }

    def _apply(self, results: Iterable[tuple[MatchContext, int, int]]) -> None:
        """Intègre une journée de résultats ; les K sont calculés avec l'état d'avant la journée."""
        results = list(results)
        if not results:
            return
        horizon = results[-1][0].match_date - timedelta(days=self.config.activity_days)
        for context, winner_id, loser_id in results:
            winner, loser = self._state(winner_id), self._state(loser_id)
            surprise = 1 - _expected(winner.elo - loser.elo)
            winner.elo += self._k(winner.matches) * surprise
            loser.elo -= self._k(loser.matches) * surprise
            if context.surface:
                self._update_surface(winner, loser, context.surface)
            for state, won in ((winner, True), (loser, False)):
                state.matches += 1
                state.recent.append(won)
                while len(state.recent) > self.config.form_window:
                    state.recent.popleft()
                state.dates.append(context.match_date)
                while state.dates[0] < horizon:  # les dates trop anciennes ne serviront plus
                    state.dates.popleft()
                state.last_date = context.match_date
            self._h2h[(winner_id, loser_id)] += 1
        self._last_day = results[-1][0].match_date

    def _update_surface(self, winner: _PlayerState, loser: _PlayerState, surface: str) -> None:
        start = self.config.elo_start
        we, le = winner.surface_elo.get(surface, start), loser.surface_elo.get(surface, start)
        wn, ln = winner.surface_matches.get(surface, 0), loser.surface_matches.get(surface, 0)
        surprise = 1 - _expected(we - le)
        winner.surface_elo[surface] = we + self._k(wn) * surprise
        loser.surface_elo[surface] = le - self._k(ln) * surprise
        winner.surface_matches[surface], loser.surface_matches[surface] = wn + 1, ln + 1

    def _peek(self, player_id: int) -> _PlayerState:
        """État d'un joueur sans le créer : un joueur jamais vu a l'état de départ (Elo initial, aucun match)."""
        return self._players.get(player_id) or _PlayerState(elo=self.config.elo_start)

    def known(self, player_id: int) -> bool:
        """Vrai si le joueur a au moins un match dans l'historique rejoué."""
        return player_id in self._players

    def ratings(self, player_id: int) -> PlayerRatings:
        """Elo général et par surface, et forme, tels que le modèle les voit pour un match à venir (lecture seule)."""
        state = self._peek(player_id)
        return PlayerRatings(
            elo=round(state.elo, 1),
            matches=state.matches,
            surface_elo={k: round(v, 1) for k, v in sorted(state.surface_elo.items())},
            surface_matches=dict(sorted(state.surface_matches.items())),
            form=round(self._form(state), 4),
            last_match=state.last_date,
        )

    def _state(self, player_id: int) -> _PlayerState:
        state = self._players.get(player_id)
        if state is None:
            state = self._players[player_id] = _PlayerState(elo=self.config.elo_start)
        return state

    def _k(self, matches: int) -> float:
        cfg = self.config
        return float(cfg.elo_k_numerator / (matches + cfg.elo_k_offset) ** cfg.elo_k_shape)

    @staticmethod
    def _form(state: _PlayerState) -> float:
        """Taux de victoire lissé sur les derniers matchs (un joueur sans historique vaut 0,5)."""
        return (sum(state.recent) + 1) / (len(state.recent) + 2)

    def _activity(self, state: _PlayerState, day: date) -> int:
        """Matchs joués dans la fenêtre précédant `day` (lecture seule : l'état n'est modifié que par `_apply`)."""
        horizon = day - timedelta(days=self.config.activity_days)
        return sum(1 for played in state.dates if played >= horizon)

    def _rest(self, state: _PlayerState, day: date) -> int:
        cap = self.config.max_rest_days
        return cap if state.last_date is None else min((day - state.last_date).days, cap)


# --------------------------------------------------------------------------- utilitaires


def _expected(elo_gap: float) -> float:
    return 1.0 / (1.0 + 10 ** (-elo_gap / 400))


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _float(value: object) -> float | None:
    return None if _missing(value) else float(value)  # type: ignore[arg-type]


def _plausible(value: float | None, low: float, high: float) -> float | None:
    return value if value is not None and low <= value <= high else None


def _int(value: object) -> int | None:
    number = _float(value)
    return None if number is None else int(number)


def _text(value: object) -> str | None:
    return None if _missing(value) or value == "" else str(value)


def _as_date(value: object) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _diff(a: float | None, b: float | None) -> float:
    return math.nan if a is None or b is None else a - b


def _log_ratio(a: float | None, b: float | None) -> float:
    return math.nan if not a or not b else math.log(a) - math.log(b)


def _log1p_diff(a: float | None, b: float | None) -> float:
    return math.nan if a is None or b is None else math.log1p(max(a, 0)) - math.log1p(max(b, 0))


def _snapshot(record: Mapping[str, Any], side: str) -> PlayerSnapshot:
    return PlayerSnapshot(
        player_id=int(record[f"{side}_id"]),
        name=_text(record.get(f"{side}_name")),
        rank=_float(record.get(f"{side}_rank")),
        rank_points=_float(record.get(f"{side}_rank_points")),
        age=_float(record.get(f"{side}_age")),
        height_cm=_plausible(_float(record.get(f"{side}_ht")), *HEIGHT_RANGE_CM),
        hand=_text(record.get(f"{side}_hand")),
    )


def _meta(record: Mapping[str, Any], context: MatchContext, p1: PlayerSnapshot, p2: PlayerSnapshot) -> dict[str, Any]:
    return {
        "match_id": str(record["match_id"]),
        "match_date": pd.Timestamp(context.match_date),
        "tourney_id": str(record["tourney_id"]),
        "tourney_name": _text(record.get("tourney_name")),
        "p1_id": p1.player_id,
        "p1_name": p1.name,
        "p1_rank": p1.rank,
        "p2_id": p2.player_id,
        "p2_name": p2.name,
        "p2_rank": p2.rank,
        "status": str(record["status"]),
    }


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    """Types stables pour la signature MLflow : flottants (NaN autorisés) et chaînes (None autorisé)."""
    for column in NUMERIC_FEATURES:
        frame[column] = frame[column].astype("float64")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("object")
    return frame
