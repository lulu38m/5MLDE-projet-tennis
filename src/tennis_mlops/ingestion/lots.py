"""Un lot d'ingestion : ce qu'on a reçu de la source, mis à plat pour les contrôles et le stockage.

Deux natures de lots :

- `tournament` : un tournoi terminé (`/tournaments/{id}/results`), avec son nombre de matchs attendu ;
- `bootstrap` : l'historique initial (`/results` sur 2010 → début du rejeu), chargé une seule fois.

La mise à plat ne corrige rien : un champ renommé ou absent reste renommé ou absent, pour que les contrôles le voient.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

import pandas as pd

from tennis_mlops.ingestion.client import SourcePayload

PLAYER_FIELDS = ("id", "name", "hand", "ht", "ioc", "age", "rank", "rank_points", "seed", "entry")
MATCH_FIELDS = (
    "match_id",
    "tourney_id",
    "tourney_name",
    "tourney_level",
    "surface",
    "draw_size",
    "best_of",
    "round",
    "match_num",
    "match_date",
    "status",
    "score",
    "minutes",
)
RESULT_COLUMNS = (*MATCH_FIELDS, *(f"{side}_{f}" for side in ("winner", "loser") for f in PLAYER_FIELDS))


class LotKind(StrEnum):
    """Nature d'un lot."""

    TOURNAMENT = "tournament"
    BOOTSTRAP = "bootstrap"


@dataclass(frozen=True, slots=True)
class Lot:
    """Un lot prêt à être contrôlé."""

    lot_id: str
    kind: LotKind
    payload: SourcePayload
    frame: pd.DataFrame
    tourney_id: str | None = None
    tourney_level: str | None = None
    end_date: date | None = None
    expected_rows: int | None = None

    @property
    def rows(self) -> int:
        """Nombre de matchs reçus."""
        return len(self.frame)

    @property
    def simulated_now(self) -> date:
        """Date simulée de la source au moment de l'appel."""
        return self.payload.simulated_now


def results_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Met à plat les résultats (`winner.id` → `winner_id`) et ajoute l'écart de classement.

    `rank_gap` n'est calculé que si les deux classements existent : il sert à la surveillance de dérive.
    """
    if not records:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))
    frame = pd.json_normalize(records, sep="_")
    if {"winner_rank", "loser_rank"} <= set(frame.columns):
        winner = pd.to_numeric(frame["winner_rank"], errors="coerce")
        loser = pd.to_numeric(frame["loser_rank"], errors="coerce")
        frame["rank_gap"] = (winner - loser).abs()
    return frame


def tournament_lot(payload: SourcePayload) -> Lot:
    """Construit un lot à partir de `GET /tournaments/{id}/results`."""
    tournament = payload.data.get("tournament", {})
    return Lot(
        lot_id=str(tournament.get("tourney_id", payload.endpoint)),
        kind=LotKind.TOURNAMENT,
        payload=payload,
        frame=results_frame(payload.data.get("results", [])),
        tourney_id=tournament.get("tourney_id"),
        tourney_level=tournament.get("tourney_level"),
        end_date=date.fromisoformat(tournament["end_date"]) if "end_date" in tournament else None,
        expected_rows=payload.data.get("expected_matches"),
    )


def bootstrap_lot(payload: SourcePayload, since: date, until: date) -> Lot:
    """Construit le lot d'amorçage à partir de `GET /results`."""
    return Lot(
        lot_id=f"bootstrap_{since.isoformat()}_{until.isoformat()}",
        kind=LotKind.BOOTSTRAP,
        payload=payload,
        frame=results_frame(payload.data.get("results", [])),
        end_date=until,
    )
