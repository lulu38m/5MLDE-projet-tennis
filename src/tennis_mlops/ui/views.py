"""Ce que l'interface affiche, calculé sans Streamlit : libellés, messages d'erreur, tableaux. Testé à part.

Les pages (`pages/`) ne font que poser ces valeurs à l'écran ; tout ce qui peut se tromper est ici.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from tennis_mlops.api.client import ApiError
from tennis_mlops.serving.history import LEVELS, SURFACES
from tennis_mlops.serving.schemas import (
    FixturePrediction,
    FixturesResponse,
    H2HResponse,
    PlayerStats,
    PlayerSummary,
    PredictResponse,
)

ROUNDS = {"F": "Finale", "SF": "Demi-finale", "QF": "Quart de finale", "R16": "Huitième de finale",
          "R32": "16e de finale", "R64": "32e de finale", "R128": "64e de finale", "RR": "Poules",
          "BR": "Match pour la 3e place", "ER": "Tour préliminaire"}  # fmt: skip
STATUS = {"upcoming": "à jouer", "awaiting_result": "résultat en attente", "played": "joué", "walkover": "forfait"}
CLAY = "#c0522b"
READINESS = {"ready": "🟢 prêt", "degraded": "🟠 dégradé", "not_ready": "🔴 pas prêt"}


@dataclass(frozen=True)
class Notice:
    """Un message à afficher : niveau, titre, explication et ce qu'il faut faire."""

    level: Literal["error", "warning", "info"]
    title: str
    text: str
    hint: str | None = None


def explain(error: ApiError, api_url: str) -> Notice:
    """Une erreur de l'API, dite pour quelqu'un qui n'a pas le code sous les yeux."""
    if error.status is None:
        return Notice("error", "API injoignable", f"Aucune réponse de {api_url}.",
                      "Démarrer la stack : `make up` (ou `make api` hors Docker), puis recharger la page.")  # fmt: skip
    if error.status == 503:
        hint = "Attendre quelques secondes : l'API recharge l'historique et le modèle."
        if "aucun modèle" in error.message or "aucune version" in error.message:
            hint = "Aucun modèle en production : lancer une campagne, `make train`."
        elif "historique" in error.message:
            hint = "Aucun match en base : lancer une ingestion, `make ingest`, ou `make demo`."
        elif "réentraîner" in error.message or "features" in error.message:
            hint = "Le modèle a été entraîné avec d'autres features : `make train`."
        return Notice("warning", "Le service ne peut pas répondre pour l'instant", error.message, hint)
    if error.kind == "unknown_player":
        return Notice("warning", "Joueur inconnu de l'historique", error.message,
                      "Un débutant peut quand même être prédit depuis la page Match, par son identifiant.")  # fmt: skip
    if error.status == 422:
        return Notice("warning", "Requête refusée par l'API", error.message)
    return Notice("error", f"Erreur de l'API ({error.status}, {error.kind})", error.message)


# ---------------------------------------------------------------------- libellés


def surface_label(surface: str | None) -> str:
    """« Terre battue » pour `Clay`."""
    return SURFACES.get(surface or "", surface or "surface inconnue")


def level_label(level: str | None) -> str:
    """« Masters 1000 » pour `M`."""
    return LEVELS.get(level or "", level or "niveau inconnu")


def round_label(round_: str | None) -> str:
    """« Quart de finale » pour `QF`."""
    return ROUNDS.get(round_ or "", round_ or "tour inconnu")


def default_best_of(level: str | None) -> int:
    """Grand Chelem au meilleur des cinq sets, le reste en trois (Coupe Davis : cinq jusqu'en 2018, trois ensuite)."""
    return 5 if level == "G" else 3


def player_label(player: PlayerSummary) -> str:
    """« Novak Djokovic (SRB) · n°1 » ; un joueur sans classement est dit tel."""
    country = f" ({player.ioc})" if player.ioc else ""
    rank = f" · n°{int(player.rank)}" if player.rank else " · non classé"
    return f"{player.name or f'joueur {player.id}'}{country}{rank}"


def percent(value: float | None, digits: int = 1) -> str:
    """0.6123 → « 61,2 % » ; rien → « — »."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:.{digits}f} %".replace(".", ",")


def number(value: float | None, digits: int = 0) -> str:
    """1234.5 → « 1 234 » ; rien → « — »."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def form_string(results: list[bool]) -> str:
    """Les derniers matchs, du plus ancien au plus récent : « V D V V »."""
    return " ".join("V" if won else "D" for won in results) or "aucun match"


# ---------------------------------------------------------------------- page Match


def comparison(
    prediction: PredictResponse, stats: tuple[PlayerStats | None, PlayerStats | None], h2h: H2HResponse | None,
    surface: str | None,
) -> pd.DataFrame:  # fmt: skip
    """Les features clés côte à côte : ce que sait l'historique de chaque joueur, et l'écart que voit le modèle."""
    p1, p2 = stats
    features = prediction.features

    def elo(s: PlayerStats | None, on: str | None) -> str:
        if s is None or s.ratings is None:
            return "1 500 (départ)"
        value = s.ratings.surface_elo.get(on, None) if on else s.ratings.elo
        return number(value if value is not None else 1500.0)

    def form(s: PlayerStats | None) -> str:
        return f"{s.form.wins} V / {s.form.losses} D" if s and s.form.results else "aucun match"

    rows = [
        ("Classement", number(prediction.player1.rank), number(prediction.player2.rank),
         _signed(features.get("rank_log_ratio"), 2, "log(rang 2 / rang 1)")),
        ("Elo général", elo(p1, None), elo(p2, None), _signed(features.get("elo_diff"), 0)),
        (f"Elo {surface_label(surface).lower()}" if surface else "Elo surface", elo(p1, surface), elo(p2, surface),
         _signed(features.get("surface_elo_diff"), 0)),
        ("Forme (10 derniers)", form(p1), form(p2), _signed(features.get("form_diff"), 2, "taux lissé")),
        ("Face-à-face", str(h2h.p1_wins if h2h else 0), str(h2h.p2_wins if h2h else 0),
         _signed(features.get("h2h_diff"), 0)),
        ("Âge", number(p1.player.age if p1 else None, 1), number(p2.player.age if p2 else None, 1),
         _signed(features.get("age_diff"), 1)),
    ]  # fmt: skip
    names = (prediction.player1.name or "Joueur 1", prediction.player2.name or "Joueur 2")
    return pd.DataFrame(rows, columns=["", names[0], names[1], "Écart vu par le modèle"]).set_index("")


def _signed(value: float | str | None, digits: int, unit: str | None = None) -> str:
    if value is None or isinstance(value, str) or math.isnan(value):
        return "—"
    text = f"{value:+.{digits}f}".replace(".", ",")
    return f"{text} ({unit})" if unit else text


# ---------------------------------------------------------------------- page Joueur


def rankings_frame(stats: PlayerStats) -> pd.DataFrame:
    """Classement et points au début de chaque tournoi."""
    rows = [{"date": pd.Timestamp(p.date), "classement": p.rank, "points": p.rank_points}
            for p in stats.ranking_history]  # fmt: skip
    return pd.DataFrame(rows, columns=["date", "classement", "points"])


def records_frame(stats: PlayerStats) -> pd.DataFrame:
    """Bilans : toutes surfaces, 52 semaines, par surface, par niveau."""
    blocks = [("Général", [stats.overall, stats.last_52_weeks]), ("Surface", stats.by_surface),
              ("Niveau", stats.by_level)]  # fmt: skip
    rows = [{"périmètre": kind, "": r.label, "victoires": r.wins, "défaites": r.losses, "% victoires": r.win_rate}
            for kind, records in blocks for r in records]  # fmt: skip
    return pd.DataFrame(rows, columns=["périmètre", "", "victoires", "défaites", "% victoires"])


def recent_frame(stats: PlayerStats) -> pd.DataFrame:
    """Derniers résultats, du plus récent au plus ancien."""
    rows = [{"date": m.match_date, "tournoi": m.tourney_name, "surface": surface_label(m.surface),
             "tour": m.round, "adversaire": m.opponent_name, "rang adv.": m.opponent_rank,
             "résultat": "✅ victoire" if m.won else "❌ défaite", "score": m.score}
            for m in stats.recent]  # fmt: skip
    columns = ["date", "tournoi", "surface", "tour", "adversaire", "rang adv.", "résultat", "score"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------- page Face-à-face


def meetings_frame(h2h: H2HResponse) -> pd.DataFrame:
    """Les rencontres, de la plus récente à la plus ancienne."""
    names = {h2h.player1.id: h2h.player1.name, h2h.player2.id: h2h.player2.name}
    rows = [{"date": m.match_date, "tournoi": m.tourney_name, "niveau": level_label(m.tourney_level),
             "surface": surface_label(m.surface), "tour": m.round, "vainqueur": names.get(m.winner_id),
             "score": m.score} for m in h2h.meetings]  # fmt: skip
    return pd.DataFrame(rows, columns=["date", "tournoi", "niveau", "surface", "tour", "vainqueur", "score"])


def h2h_surfaces_frame(h2h: H2HResponse) -> pd.DataFrame:
    """Bilan par surface."""
    rows = [{"surface": surface_label(s.surface), h2h.player1.name or "joueur 1": s.p1_wins,
             h2h.player2.name or "joueur 2": s.p2_wins} for s in h2h.by_surface]  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- page Matchs à venir


def fixtures_frames(response: FixturesResponse) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(à venir ou en attente du résultat, triés par date ; joués, du plus récent au plus ancien)."""
    upcoming = [f for f in response.fixtures if f.status in {"upcoming", "awaiting_result"}]
    played = [f for f in response.fixtures if f.status in {"played", "walkover"}][::-1]
    return (pd.DataFrame([_fixture_row(f) for f in upcoming], columns=UPCOMING_COLUMNS),
            pd.DataFrame([_played_row(f) for f in played], columns=PLAYED_COLUMNS))  # fmt: skip


UPCOMING_COLUMNS = ["date", "tournoi", "tour", "joueur 1", "proba 1", "joueur 2", "favori", "version", "état"]
PLAYED_COLUMNS = ["date", "tournoi", "tour", "joueur 1", "joueur 2", "favori", "vainqueur", "proba du vainqueur",
                  "le modèle avait", "score", "version"]  # fmt: skip


def _fixture_row(f: FixturePrediction) -> dict[str, object]:
    return {"date": f.match_date, "tournoi": f.tourney_name, "tour": f.round, "joueur 1": f.player1.name,
            "proba 1": f.player1.win_probability, "joueur 2": f.player2.name, "favori": _name(f, f.predicted_winner_id),
            "version": f"v{f.model_version}", "état": STATUS[f.status]}  # fmt: skip


def _played_row(f: FixturePrediction) -> dict[str, object]:
    verdict = "forfait, non jugé" if f.correct is None else ("✅ raison" if f.correct else "❌ tort")
    return {"date": f.match_date, "tournoi": f.tourney_name, "tour": f.round, "joueur 1": f.player1.name,
            "joueur 2": f.player2.name, "favori": _name(f, f.predicted_winner_id),
            "vainqueur": _name(f, f.actual_winner_id), "proba du vainqueur": f.winner_probability,
            "le modèle avait": verdict, "score": f.score, "version": f"v{f.model_version}"}  # fmt: skip


def _name(f: FixturePrediction, player_id: int | None) -> str | None:
    if player_id is None:
        return None
    return f.player1.name if player_id == f.player1.id else f.player2.name
