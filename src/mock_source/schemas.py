"""Contrats JSON de l'API de rejeu (validés et documentés automatiquement dans `/docs`)."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from tennis_mlops.data.score import MatchStatus


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlayerEntry(_Schema):
    """Un joueur tel qu'inscrit dans un match (informations connues avant le match)."""

    id: int
    name: str
    hand: str | None = None
    ht: int | None = Field(default=None, description="Taille en cm")
    ioc: str | None = Field(default=None, description="Pays (code CIO)")
    age: float | None = None
    rank: int | None = Field(default=None, description="Classement au début du tournoi ; null = non classé")
    rank_points: int | None = None
    seed: int | None = None
    entry: str | None = Field(default=None, description="Q, WC, LL, PR…")


class MatchInfo(_Schema):
    """Champs communs aux matchs à venir et terminés."""

    match_id: str = Field(description="Clé stable `{tourney_id}-{match_num}`, utilisée pour le feedback")
    tourney_id: str
    tourney_name: str
    tourney_level: str
    surface: str | None
    draw_size: int
    best_of: int
    round: str
    match_num: int
    match_date: dt.date = Field(description="Date estimée : début du tournoi + jour selon le tour")


class Fixture(MatchInfo):
    """Match à venir : aucun résultat, joueurs dans un ordre neutre (hash du match_id)."""

    player1: PlayerEntry
    player2: PlayerEntry


class MatchResult(MatchInfo):
    """Match terminé."""

    winner: PlayerEntry
    loser: PlayerEntry
    score: str | None
    minutes: int | None
    status: MatchStatus


class ResultsPage(_Schema):
    """Réponse de `GET /results`."""

    since: dt.date
    until: dt.date
    count: int
    results: list[MatchResult]
    reason: str | None = Field(default=None, description="Explication quand la période est vide")


class FixturesPage(_Schema):
    """Réponse de `GET /fixtures`."""

    now: dt.date
    horizon_days: int
    count: int
    fixtures: list[Fixture]
    reason: str | None = None


class TournamentStatus(StrEnum):
    """Avancement d'un tournoi par rapport à l'horloge simulée."""

    UPCOMING = "upcoming"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TournamentSummary(_Schema):
    """Un tournoi du calendrier, vu à la date simulée."""

    tourney_id: str
    tourney_name: str
    tourney_level: str
    surface: str | None
    draw_size: int
    start_date: dt.date
    end_date: dt.date
    status: TournamentStatus
    expected_matches: int = Field(description="Nombre total de matchs du tournoi (pour le contrôle de volume)")
    completed_matches: int


class TournamentsPage(_Schema):
    """Réponse de `GET /tournaments`."""

    count: int
    tournaments: list[TournamentSummary]


class TournamentResults(_Schema):
    """Réponse de `GET /tournaments/{id}/results` : un lot d'ingestion."""

    tournament: TournamentSummary
    expected_matches: int
    count: int
    results: list[MatchResult]


class RankingEntry(_Schema):
    """Une ligne de classement."""

    rank: int
    player_id: int
    name: str | None
    points: int | None


class RankingsPage(_Schema):
    """Réponse de `GET /rankings`."""

    requested_date: dt.date
    ranking_date: dt.date | None = Field(description="Dernier classement publié à la date demandée")
    count: int
    rankings: list[RankingEntry]
    reason: str | None = None


class PlayerProfile(_Schema):
    """Fiche d'un joueur."""

    id: int
    name: str
    hand: str | None = None
    dob: dt.date | None = None
    ioc: str | None = None
    height: int | None = None


class RankingPoint(_Schema):
    """Classement d'un joueur une semaine donnée."""

    ranking_date: dt.date
    rank: int
    points: int | None


class PlayerMatch(_Schema):
    """Un match du point de vue d'un joueur."""

    match_id: str
    match_date: dt.date
    tourney_id: str
    tourney_name: str
    tourney_level: str
    surface: str | None
    round: str
    opponent_id: int
    opponent_name: str
    won: bool
    score: str | None
    status: MatchStatus


class PlayerHistory(_Schema):
    """Réponse de `GET /players/{id}/history`."""

    player: PlayerProfile
    as_of: dt.date
    is_debutant: bool = Field(description="Aucun match dans l'historique avant as_of")
    wins: int
    losses: int
    rankings: list[RankingPoint]
    matches: list[PlayerMatch]


class ClockRequest(_Schema):
    """Corps de `POST /admin/clock` : au moins un des deux champs."""

    date: dt.date | None = None
    speed: float | None = Field(default=None, ge=0, le=1000, description="Jours simulés par minute réelle")


class AdvanceRequest(_Schema):
    """Corps de `POST /admin/clock/advance`."""

    days: int = Field(default=7, ge=1, le=366)


class ClockState(_Schema):
    """État de l'horloge simulée."""

    now: dt.date
    speed: float
    lower: dt.date
    upper: dt.date


class AdvanceResponse(_Schema):
    """Réponse de `POST /admin/clock/advance`."""

    previous: dt.date
    now: dt.date
    newly_completed_matches: int
    newly_completed_tournaments: list[str]


class Health(_Schema):
    """Réponse de `GET /health`."""

    status: str
    dataset_version: str
    simulated_now: dt.date
    matches_loaded: int
    chaos_mode: str
