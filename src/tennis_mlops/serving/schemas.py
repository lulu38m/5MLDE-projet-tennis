"""Contrat de l'API de prédiction : ce qu'on envoie, ce qu'on reçoit. Partagé par l'API, ses clients et les tests.

Les valeurs autorisées (surface, niveau, tour, main) sont celles des données Sackmann : une faute de frappe donne un
`422` explicite plutôt qu'une catégorie inconnue que le modèle ignorerait sans rien dire.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Surface = Literal["Hard", "Clay", "Grass", "Carpet"]
Level = Literal["G", "M", "A", "F", "D", "O"]
Round = Literal["R128", "R64", "R32", "R16", "QF", "SF", "F", "RR", "BR", "ER"]
Hand = Literal["R", "L", "U", "A"]
MATCH_ID_PATTERN = r"^[\w.-]{1,64}$"
BATCH_LIMIT = 1000  # plafond du schéma ; la limite appliquée vient de configs/project.yaml (api.max_batch)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlayerInput(_Strict):
    """Un joueur : son identifiant suffit ; le reste remplace le dernier profil connu (classement du tournoi…)."""

    id: int = Field(gt=0, description="Identifiant Sackmann (player_id)")
    rank: int | None = Field(default=None, ge=1, le=5000, description="Classement au début du tournoi")
    rank_points: int | None = Field(default=None, ge=0, le=100_000)
    age: float | None = Field(default=None, ge=14, le=50, description="Âge en années, au jour du match")
    height_cm: int | None = Field(default=None, ge=150, le=230)
    hand: Hand | None = None


class MatchRequest(_Strict):
    """Un match à venir."""

    match_id: str | None = Field(
        default=None, pattern=MATCH_ID_PATTERN, description="Clé du feedback (`{tourney_id}-{match_num}`)"
    )
    match_date: dt.date | None = Field(default=None, description="Défaut : le lendemain du dernier match connu")
    surface: Surface | None = None
    tourney_level: Level | None = Field(default=None, description="G Grand Chelem, M Masters, A ATP 250/500…")
    round: Round | None = None
    best_of: Literal[3, 5] | None = None
    tourney_name: str | None = Field(default=None, max_length=120, description="Affiché par l'interface")
    player1: PlayerInput
    player2: PlayerInput

    @model_validator(mode="after")
    def _two_players(self) -> Self:
        if self.player1.id == self.player2.id:
            msg = "player1 et player2 doivent être deux joueurs différents"
            raise ValueError(msg)
        return self


class BatchRequest(_Strict):
    """Plusieurs matchs en un appel (scoring des matchs de la semaine)."""

    matches: list[MatchRequest] = Field(min_length=1, max_length=BATCH_LIMIT)


class ModelRef(BaseModel):
    """La version qui a produit la prédiction."""

    name: str
    version: str
    run_id: str
    alias: str = "production"


class PlayerOutput(BaseModel):
    """Un joueur dans la réponse."""

    id: int
    name: str | None
    win_probability: float
    rank: float | None = Field(description="Classement utilisé (requête, sinon dernier connu)")
    known: bool = Field(description="Au moins un match dans l'historique")


class Prediction(BaseModel):
    """La prédiction d'un match."""

    prediction_id: str
    match_id: str | None
    match_date: dt.date
    player1: PlayerOutput
    player2: PlayerOutput
    predicted_winner_id: int
    features: dict[str, float | str | None]
    features_as_of: dt.date | None = Field(description="Dernière journée de l'historique utilisé")
    warnings: list[str] = []


class PredictResponse(Prediction):
    """`POST /predict`."""

    request_id: str
    model: ModelRef
    latency_ms: float


class BatchResponse(BaseModel):
    """`POST /predict/batch`."""

    request_id: str
    model: ModelRef
    count: int
    latency_ms: float
    predictions: list[Prediction]


class FeedbackRequest(_Strict):
    """Le vrai résultat d'un match prédit."""

    match_id: str = Field(pattern=MATCH_ID_PATTERN)
    winner_id: int = Field(gt=0)


class FeedbackOutcome(BaseModel):
    """Une prédiction de ce match, maintenant jugée."""

    prediction_id: str
    model_version: str
    predicted_winner_id: int
    winner_probability: float = Field(description="Probabilité que la prédiction donnait au vrai vainqueur")
    correct: bool
    log_loss: float


class FeedbackResponse(BaseModel):
    """`POST /feedback`."""

    request_id: str
    match_id: str
    winner_id: int
    updated: bool = Field(description="Un résultat était déjà enregistré pour ce match (remplacé)")
    predictions: list[FeedbackOutcome]


# ---------------------------------------------------------------------- consultation (interface de test, phase 8)


class PlayerSummary(BaseModel):
    """Un joueur tel que l'historique le connaît à une date (jamais après la dernière journée intégrée)."""

    id: int
    name: str | None
    ioc: str | None = None
    hand: str | None = None
    height_cm: float | None = None
    rank: float | None = Field(default=None, description="Dernier classement connu à cette date")
    rank_points: float | None = None
    age: float | None = Field(default=None, description="Âge à cette date")
    last_match: dt.date | None = None
    matches: int = Field(default=0, description="Matchs joués dans l'historique (forfaits exclus)")


class PlayersResponse(BaseModel):
    """`GET /players`."""

    query: str | None
    as_of: dt.date | None
    count: int
    players: list[PlayerSummary]


class Record(BaseModel):
    """Victoires et défaites sur un périmètre (toutes, une surface, un niveau de tournoi)."""

    key: str
    label: str
    wins: int
    losses: int
    win_rate: float | None


class Ratings(BaseModel):
    """Elo et forme tels que le modèle les verrait pour un match du lendemain de l'historique."""

    elo: float
    matches: int
    surface_elo: dict[str, float]
    surface_matches: dict[str, int]
    form: float = Field(description="Taux de victoire lissé sur les derniers matchs (feature `form_diff`)")


class FormSummary(BaseModel):
    """Les derniers matchs, du plus ancien au plus récent."""

    window: int
    wins: int
    losses: int
    results: list[bool]


class RankingPoint(BaseModel):
    """Classement publié au début d'un tournoi."""

    date: dt.date
    rank: float | None
    rank_points: float | None


class MatchLine(BaseModel):
    """Un match d'un joueur, vu de son côté."""

    match_id: str
    match_date: dt.date
    tourney_name: str | None
    tourney_level: str | None
    surface: str | None
    round: str | None
    opponent_id: int
    opponent_name: str | None
    opponent_rank: float | None
    won: bool
    score: str | None


class PlayerStats(BaseModel):
    """`GET /players/{id}/stats`."""

    player: PlayerSummary
    as_of: dt.date = Field(description="Date des statistiques (bornée à la dernière journée de l'historique)")
    history_last_day: dt.date | None
    ratings: Ratings | None = Field(description="Uniquement à la dernière journée : l'Elo n'est pas historisé")
    overall: Record
    last_52_weeks: Record
    by_surface: list[Record]
    by_level: list[Record]
    form: FormSummary
    ranking_history: list[RankingPoint]
    recent: list[MatchLine]


class PlayerRef(BaseModel):
    """Identifiant et nom."""

    id: int
    name: str | None


class H2HMeeting(BaseModel):
    """Une rencontre entre les deux joueurs."""

    match_id: str
    match_date: dt.date
    tourney_name: str | None
    tourney_level: str | None
    surface: str | None
    round: str | None
    winner_id: int
    score: str | None


class H2HSurface(BaseModel):
    """Bilan des rencontres sur une surface."""

    surface: str
    p1_wins: int
    p2_wins: int


class H2HResponse(BaseModel):
    """`GET /h2h`."""

    player1: PlayerRef
    player2: PlayerRef
    as_of: dt.date | None
    total: int
    p1_wins: int
    p2_wins: int
    by_surface: list[H2HSurface]
    meetings: list[H2HMeeting]


FixtureStatus = Literal["upcoming", "awaiting_result", "played", "walkover"]


class FixturePlayer(BaseModel):
    """Un joueur d'un match prédit."""

    id: int
    name: str | None
    win_probability: float


class FixturePrediction(BaseModel):
    """Un match prédit à l'avance, et son résultat une fois validé en base."""

    match_id: str
    match_date: dt.date
    tourney_name: str | None
    tourney_level: str | None
    surface: str | None
    round: str | None
    player1: FixturePlayer
    player2: FixturePlayer
    predicted_winner_id: int
    model_version: str
    predicted_at: dt.datetime
    client: str | None
    status: FixtureStatus = Field(
        description="upcoming (à jouer), awaiting_result (date passée, lot pas encore validé), played, walkover"
    )
    actual_winner_id: int | None = None
    score: str | None = None
    correct: bool | None = None
    winner_probability: float | None = Field(default=None, description="Probabilité donnée au vrai vainqueur")


class FixturesResponse(BaseModel):
    """`GET /fixtures/predictions`."""

    as_of: dt.date | None = Field(description="Dernière journée de l'historique")
    watermark: dt.date | None = Field(description="Horloge simulée vue par la dernière ingestion")
    since: dt.date
    client: str | None
    count: int
    played: int
    correct: int
    accuracy: float | None
    fixtures: list[FixturePrediction]
