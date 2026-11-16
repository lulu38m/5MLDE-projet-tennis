"""Les features du labo, calculées en rejouant l'historique match par match, sans jamais regarder le futur.

Les résultats d'une journée n'entrent dans l'état des joueurs qu'à la fin de cette journée (date estimée du match,
un tour par jour) : un match ne voit que ce qui était connu la veille. Le joueur 1 est tiré par le hash du
`match_id`, comme dans le pipeline, pour que la cible ne soit pas toujours « le premier gagne ».

Familles (préfixe `a_` / `b_` pour chaque joueur, `d_` pour l'écart a - b) :

- **Elo** : général, par surface, mélangé, et « avec marge » (le K est pondéré par la part de jeux gagnés) ;
- **service / retour** : points gagnés au service et en retour, 1re balle, aces, doubles fautes, balles de
  break sauvées et converties, en moyenne exponentielle (générale et par surface), ramenées vers la moyenne du
  circuit quand l'historique est court ; puis la **chaîne de Markov** point → jeu → set → match ;
- **forme** : 10 derniers matchs, moyenne exponentielle, résultat contre l'attendu Elo ;
- **fatigue** : jours de repos, matchs et minutes sur 7 / 30 jours, minutes déjà jouées dans le tournoi ;
- **expérience** : matchs joués, sur la surface, au niveau, bilan en 5 sets et en set décisif, passé au tournoi ;
- **classement** : log du rang et des points, évolution sur 3 mois, meilleur rang ;
- **profil** : âge, taille, gaucher, statut d'entrée (qualifié, invité, lucky loser), tête de série ;
- **face-à-face** : général et sur la surface ; **contexte** : surface, niveau, tour, format, taille du tableau.
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from lab import markov
from mock_source.repository import player1_is_winner

PLAYED = frozenset({"completed", "retired"})
ROUND_ORDER = {"RR": 0, "ER": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "BR": 6, "F": 7}
SURFACES = ("Hard", "Clay", "Grass", "Carpet")
STAT_KEYS = (
    "spw",
    "svpt",
    "rpw",
    "rpt",
    "first_in",
    "aces",
    "dfs",
    "bp_faced",
    "bp_saved",
    "bp_chances",
    "bp_won",
    "spw_adj",
    "rpw_adj",
)
_SET = re.compile(r"(\d+)-(\d+)")


@dataclass(frozen=True)
class FeatureParams:
    """Réglages du rejeu (versionnés dans `lab/config.yaml`, tracés dans MLflow)."""

    elo_start: float = 1500.0
    k_numerator: float = 250.0
    k_offset: float = 5.0
    k_shape: float = 0.4
    surface_weight: float = 0.5
    mov_weight: float = 1.0  # Elo « avec marge » : K x (1 + mov_weight x (part de jeux - 0,5) x 2)
    stats_half_life: float = 40.0  # en matchs : poids divisé par deux tous les 40 matchs
    surface_stats_half_life: float = 20.0
    form_half_life: float = 15.0
    shrink_points: float = 300.0  # pseudo-points vers la moyenne du circuit
    retired_k_factor: float = 0.5  # un abandon compte moitié pour l'Elo


def _k(params: FeatureParams, n: int) -> float:
    return float(params.k_numerator / (n + params.k_offset) ** params.k_shape)


def expected(a: float, b: float) -> float:
    """Probabilité Elo que A batte B."""
    return float(1.0 / (1.0 + 10 ** ((b - a) / 400)))


def games_share(score: str | None) -> float | None:
    """Part des jeux gagnés par le vainqueur (`6-4 7-6(5)` → 13/23) ; None si illisible."""
    if not score:
        return None
    won = lost = 0
    for a, b in _SET.findall(re.sub(r"\(\d+\)|\[.*?\]", "", score)):
        won, lost = won + int(a), lost + int(b)
    return won / (won + lost) if won + lost else None


def sets_played(score: str | None) -> int:
    """Nombre de sets du score."""
    return len(_SET.findall(re.sub(r"\(\d+\)|\[.*?\]", "", score or "")))


class _Ew:
    """Somme exponentielle de plusieurs compteurs : chaque nouveau match divise l'ancien poids par 2^(1/demi-vie)."""

    __slots__ = ("decay", "sums")

    def __init__(self, half_life: float) -> None:
        self.decay = 0.5 ** (1 / half_life)
        self.sums = dict.fromkeys(STAT_KEYS, 0.0)

    def add(self, values: dict[str, float]) -> None:
        for key in STAT_KEYS:
            self.sums[key] = self.sums[key] * self.decay + values[key]


@dataclass
class _Player:
    params: FeatureParams
    elo: float = 1500.0
    mov: float = 1500.0
    n: int = 0
    surface_elo: dict[str, float] = field(default_factory=dict)
    surface_n: dict[str, int] = field(default_factory=dict)
    level_n: dict[str, int] = field(default_factory=dict)
    stats: _Ew = field(init=False)
    surface_stats: dict[str, _Ew] = field(default_factory=dict)
    results: deque[tuple[bool, float]] = field(default_factory=lambda: deque(maxlen=20))  # (gagné, attendu)
    form_ew: float = 0.5
    activity: deque[tuple[date, float, str]] = field(default_factory=lambda: deque(maxlen=30))
    last_date: date | None = None
    bo5: list[int] = field(default_factory=lambda: [0, 0])
    deciding: list[int] = field(default_factory=lambda: [0, 0])
    tourneys: dict[str, list[int]] = field(default_factory=dict)
    ranks: deque[tuple[date, float]] = field(default_factory=lambda: deque(maxlen=60))
    elo_hist: deque[tuple[date, float]] = field(default_factory=lambda: deque(maxlen=80))
    year: deque[tuple[date, bool]] = field(default_factory=lambda: deque(maxlen=120))
    best_rank: float = math.inf

    def __post_init__(self) -> None:
        self.elo = self.mov = self.params.elo_start
        self.stats = _Ew(self.params.stats_half_life)

    def s_elo(self, surface: str | None) -> float:
        return self.surface_elo.get(surface or "", self.params.elo_start)


class FeatureEngine:
    """Rejoue les matchs dans l'ordre et produit une ligne de features par match joué."""

    def __init__(self, params: FeatureParams | None = None) -> None:
        self.params = params or FeatureParams()
        self.players: dict[int, _Player] = {}
        self.h2h: dict[tuple[int, int], int] = {}
        self.h2h_surface: dict[tuple[int, int, str], int] = {}
        self.tour_spw = 0.63  # moyenne du circuit (points gagnés au service), mise à jour au fil de l'eau
        self._tour = [0.0, 0.0]

    def player(self, pid: int) -> _Player:
        """L'état d'un joueur (créé à sa première apparition)."""
        if pid not in self.players:
            self.players[pid] = _Player(self.params)
        return self.players[pid]

    # ------------------------------------------------------------------ rejeu

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Une ligne par match joué (terminé ou abandon), dans l'ordre chronologique."""
        rows: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        current: date | None = None
        for record in self._records(matches):
            day = record["match_date"]
            if day != current:
                for done in pending:
                    self._update(done)
                pending, current = [], day
            if record["status"] not in PLAYED:
                continue
            rows.append(self._row(record))
            pending.append(record)
        for done in pending:
            self._update(done)
        return pd.DataFrame(rows)

    @staticmethod
    def _records(matches: pd.DataFrame) -> Iterator[dict[str, Any]]:
        ordered = matches.assign(_round=matches["round"].map(ROUND_ORDER).fillna(0)).sort_values(
            ["match_date", "tourney_id", "_round", "match_num"], kind="stable"
        )
        for record in ordered.to_dict("records"):
            yield {str(k): (None if isinstance(v, float) and math.isnan(v) else v) for k, v in record.items()}

    # ------------------------------------------------------------------ features

    def _row(self, r: dict[str, Any]) -> dict[str, Any]:
        winner_first = player1_is_winner(str(r["match_id"]))
        a_side, b_side = ("winner", "loser") if winner_first else ("loser", "winner")
        a_id, b_id = int(r[f"{a_side}_id"]), int(r[f"{b_side}_id"])
        a, b = self.player(a_id), self.player(b_id)
        surface, day, best_of = r.get("surface"), r["match_date"], int(r.get("best_of") or 3)
        row: dict[str, Any] = {
            "match_id": r["match_id"], "match_date": day, "tourney_id": r["tourney_id"],
            "tourney_name": r.get("tourney_name"), "surface": surface or "inconnu",
            "tourney_level": r.get("tourney_level") or "inconnu", "round": r.get("round") or "inconnu",
            "round_order": ROUND_ORDER.get(r.get("round") or "", 0), "best_of": best_of,
            "draw_size": r.get("draw_size") or 0, "status": r["status"], "p1_id": a_id, "p2_id": b_id,
            "p1_name": r.get(f"{a_side}_name"), "p2_name": r.get(f"{b_side}_name"), "label": int(winner_first),
        }  # fmt: skip
        sides = {"a": self._player_features(a, r, a_side, surface, day), "b": self._player_features(b, r, b_side,
                 surface, day)}  # fmt: skip
        for key in sides["a"]:
            va, vb = sides["a"][key], sides["b"][key]
            row[f"a_{key}"], row[f"b_{key}"] = va, vb
            row[f"d_{key}"] = va - vb if va is not None and vb is not None else None
        w = self.params.surface_weight
        row["elo_prob"] = expected(a.elo, b.elo)
        row["surface_elo_prob"] = expected(a.s_elo(surface), b.s_elo(surface))
        row["blend_elo_prob"] = expected((1 - w) * a.elo + w * a.s_elo(surface), (1 - w) * b.elo + w * b.s_elo(surface))
        row["mov_elo_prob"] = expected(a.mov, b.mov)
        row["markov_prob"] = self._markov(row, "spw", "rpw", best_of)
        row["markov_surface_prob"] = self._markov(row, "s_spw", "s_rpw", best_of)
        row["markov_adj_prob"] = self._markov(row, "spw_adj", "rpw_adj", best_of)
        row["h2h_n"] = self.h2h.get((a_id, b_id), 0) + self.h2h.get((b_id, a_id), 0)
        row["d_h2h"] = self.h2h.get((a_id, b_id), 0) - self.h2h.get((b_id, a_id), 0)
        key = surface or ""
        row["d_h2h_surface"] = self.h2h_surface.get((a_id, b_id, key), 0) - self.h2h_surface.get((b_id, a_id, key), 0)
        row["lefty_vs_righty"] = int(row["a_lefty"] != row["b_lefty"])
        return row

    def _markov(self, row: dict[str, Any], serve: str, ret: str, best_of: int) -> float:
        """Points au service de A contre B : son service + (retour moyen - retour de B) (Barnett et Clarke)."""
        avg_return = 1 - self.tour_spw
        pa = row[f"a_{serve}"] + (avg_return - row[f"b_{ret}"])
        pb = row[f"b_{serve}"] + (avg_return - row[f"a_{ret}"])
        return markov.match(pa, pb, best_of)

    def _player_features(
        self, p: _Player, r: dict[str, Any], side: str, surface: str | None, day: date
    ) -> dict[str, Any]:
        params, tour = self.params, self.tour_spw
        st, sst = p.stats.sums, p.surface_stats.get(surface or "")
        shrink = params.shrink_points

        def rate(num: float, den: float, prior: float) -> float:
            return (num + prior * shrink) / (den + shrink)

        s_sums = sst.sums if sst else dict.fromkeys(STAT_KEYS, 0.0)
        recent = [(d, m, t) for d, m, t in p.activity if (day - d).days <= 30]
        rank = r.get(f"{side}_rank")
        three_months = [rk for d, rk in p.ranks if 80 <= (day - d).days <= 120]
        age = r.get(f"{side}_age")
        results = list(p.results)
        last10 = results[-10:]
        tourney = p.tourneys.get(str(r.get("tourney_name")), [0, 0])
        level = r.get("tourney_level") or ""
        entry = str(r.get(f"{side}_entry") or "")
        return {
            "elo": p.elo, "s_elo": p.s_elo(surface), "mov_elo": p.mov, "n": p.n,
            "s_n": p.surface_n.get(surface or "", 0), "level_n": p.level_n.get(level, 0),
            "spw": rate(st["spw"], st["svpt"], tour), "rpw": rate(st["rpw"], st["rpt"], 1 - tour),
            "s_spw": rate(s_sums["spw"], s_sums["svpt"], rate(st["spw"], st["svpt"], tour)),
            "s_rpw": rate(s_sums["rpw"], s_sums["rpt"], rate(st["rpw"], st["rpt"], 1 - tour)),
            "first_in": rate(st["first_in"], st["svpt"], 0.61), "ace_rate": rate(st["aces"], st["svpt"], 0.07),
            "df_rate": rate(st["dfs"], st["svpt"], 0.035),
            "bp_save": (st["bp_saved"] + 0.6 * 30) / (st["bp_faced"] + 30),
            "bp_conv": (st["bp_won"] + 0.4 * 30) / (st["bp_chances"] + 30),
            "form10": sum(w for w, _ in last10) / len(last10) if last10 else 0.5,
            "form_ew": p.form_ew,
            "spw_adj": rate(st["spw_adj"], st["svpt"], tour), "rpw_adj": rate(st["rpw_adj"], st["rpt"], 1 - tour),
            "elo_trend": p.elo - next((e for d, e in p.elo_hist if (day - d).days <= 90), p.elo),
            "matches_52w": sum(1 for d, _ in p.year if (day - d).days <= 364),
            "win_52w": (sum(w for d, w in p.year if (day - d).days <= 364) + 1)
            / (sum(1 for d, _ in p.year if (day - d).days <= 364) + 2),
            "over_expected": sum(w - e for w, e in results),
            "rest_days": min((day - p.last_date).days, 120) if p.last_date else 120,
            "matches_7d": sum(1 for d, _, _ in recent if (day - d).days <= 7), "matches_30d": len(recent),
            "minutes_7d": sum(m for d, m, _ in recent if (day - d).days <= 7),
            "minutes_tourney": sum(m for _, m, t in recent if t == r["tourney_id"]),
            "bo5_rate": (p.bo5[0] + 1) / (p.bo5[0] + p.bo5[1] + 2),
            "deciding_rate": (p.deciding[0] + 1) / (p.deciding[0] + p.deciding[1] + 2),
            "tourney_wins": tourney[0], "tourney_rate": (tourney[0] + 1) / (tourney[0] + tourney[1] + 2),
            "log_rank": math.log(rank) if rank else math.log(2000),
            "log_points": math.log1p(r.get(f"{side}_rank_points") or 0),
            "rank_trend": (math.log(np.mean(three_months)) - math.log(rank)) if three_months and rank else 0.0,
            "log_best_rank": math.log(p.best_rank) if p.best_rank < math.inf else math.log(2000),
            "age": age if age is not None else 26.0, "height": r.get(f"{side}_ht") or 185.0,
            "lefty": int(r.get(f"{side}_hand") == "L"), "qualifier": int(entry in {"Q", "LL"}),
            "wildcard": int(entry == "WC"), "seeded": int(r.get(f"{side}_seed") is not None),
        }  # fmt: skip

    # ------------------------------------------------------------------ mise à jour, en fin de journée

    def _update(self, r: dict[str, Any]) -> None:
        params = self.params
        w_id, l_id = int(r["winner_id"]), int(r["loser_id"])
        w, lo = self.player(w_id), self.player(l_id)
        surface, day, level = r.get("surface") or "", r["match_date"], r.get("tourney_level") or ""
        factor = params.retired_k_factor if r["status"] == "retired" else 1.0
        exp_w = expected(w.elo, lo.elo)
        k_w, k_l = _k(params, w.n) * factor, _k(params, lo.n) * factor
        w.elo, lo.elo = w.elo + k_w * (1 - exp_w), lo.elo - k_l * (1 - exp_w)
        es = expected(w.s_elo(surface), lo.s_elo(surface))
        ks_w, ks_l = _k(params, w.surface_n.get(surface, 0)) * factor, _k(params, lo.surface_n.get(surface, 0)) * factor
        w.surface_elo[surface] = w.s_elo(surface) + ks_w * (1 - es)
        lo.surface_elo[surface] = lo.s_elo(surface) - ks_l * (1 - es)
        share = games_share(r.get("score"))
        boost = 1 + params.mov_weight * ((share - 0.5) * 2 if share is not None else 0)
        em = expected(w.mov, lo.mov)
        w.mov, lo.mov = w.mov + k_w * boost * (1 - em), lo.mov - k_l * boost * (1 - em)
        for p, won, exp in ((w, True, exp_w), (lo, False, 1 - exp_w)):
            p.n += 1
            p.surface_n[surface] = p.surface_n.get(surface, 0) + 1
            p.level_n[level] = p.level_n.get(level, 0) + 1
            p.results.append((won, exp))
            decay = 0.5 ** (1 / params.form_half_life)
            p.form_ew = p.form_ew * decay + (1 - decay) * won
            p.activity.append((day, float(r.get("minutes") or 0), str(r["tourney_id"])))
            p.year.append((day, won))
            p.elo_hist.append((day, p.elo))
            p.last_date = day
            if int(r.get("best_of") or 3) == 5:
                p.bo5[0 if won else 1] += 1
            if sets_played(r.get("score")) in (3, 5) and r["status"] == "completed":
                deciding = sets_played(r.get("score")) == (5 if int(r.get("best_of") or 3) == 5 else 3)
                if deciding:
                    p.deciding[0 if won else 1] += 1
            hist = p.tourneys.setdefault(str(r.get("tourney_name")), [0, 0])
            hist[0 if won else 1] += 1
        for p, side in ((w, "winner"), (lo, "loser")):
            rank = r.get(f"{side}_rank")
            if rank:
                p.ranks.append((day, float(rank)))
                p.best_rank = min(p.best_rank, float(rank))
            while p.ranks and (day - p.ranks[0][0]) > timedelta(days=130):
                p.ranks.popleft()
        self.h2h[(w_id, l_id)] = self.h2h.get((w_id, l_id), 0) + 1
        self.h2h_surface[(w_id, l_id, surface)] = self.h2h_surface.get((w_id, l_id, surface), 0) + 1
        self._update_stats(r, w, lo, surface)

    def _update_stats(self, r: dict[str, Any], w: _Player, lo: _Player, surface: str) -> None:
        if r.get("w_svpt") in (None, 0) or r.get("l_svpt") in (None, 0):
            return
        sides = {"w": r, "l": r}

        def side(s: str, o: str) -> dict[str, float]:
            def g(c: str, x: str = s) -> float:
                return float(sides[x].get(f"{x}_{c}") or 0)

            spw = g("1stWon") + g("2ndWon")
            opp_spw = g("1stWon", o) + g("2ndWon", o)
            return {
                "spw": spw, "svpt": g("svpt"), "rpw": g("svpt", o) - opp_spw, "rpt": g("svpt", o),
                "first_in": g("1stIn"), "aces": g("ace"), "dfs": g("df"), "bp_faced": g("bpFaced"),
                "bp_saved": g("bpSaved"), "bp_chances": g("bpFaced", o), "bp_won": g("bpFaced", o) - g("bpSaved", o),
            }  # fmt: skip

        tour = self.tour_spw
        shrink = self.params.shrink_points

        def before(p: _Player, num: str, den: str, prior: float) -> float:
            return (p.stats.sums[num] + prior * shrink) / (p.stats.sums[den] + shrink)

        opp_rpw = {"w": before(lo, "rpw", "rpt", 1 - tour), "l": before(w, "rpw", "rpt", 1 - tour)}
        opp_spw = {"w": before(lo, "spw", "svpt", tour), "l": before(w, "spw", "svpt", tour)}
        for p, s, o in ((w, "w", "l"), (lo, "l", "w")):
            values = side(s, o)
            # Points gagnés corrigés de la force de l'adversaire : mieux servir contre un bon relanceur compte plus.
            values["spw_adj"] = values["spw"] + (opp_rpw[s] - (1 - tour)) * values["svpt"]
            values["rpw_adj"] = values["rpw"] + (opp_spw[s] - tour) * values["rpt"]
            if values["svpt"] <= 0 or values["spw"] > values["svpt"]:
                continue
            p.stats.add(values)
            p.surface_stats.setdefault(surface, _Ew(self.params.surface_stats_half_life)).add(values)
            self._tour[0] += values["spw"]
            self._tour[1] += values["svpt"]
        if self._tour[1] > 50_000:
            self.tour_spw = self._tour[0] / self._tour[1]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Colonnes numériques utilisables par un modèle (tout sauf les identifiants, la cible et les catégories)."""
    excluded = {"match_id", "match_date", "tourney_id", "tourney_name", "surface", "tourney_level", "round", "status",
                "p1_id", "p2_id", "p1_name", "p2_name", "label"}  # fmt: skip
    return [c for c in frame.columns if c not in excluded]


CATEGORICAL = ("surface", "tourney_level", "round")
