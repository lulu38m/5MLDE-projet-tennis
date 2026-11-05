"""Estimation du jour de chaque match dans son tournoi.

Les fichiers Sackmann ne donnent que la date de début du tournoi (`tourney_date`) et le tour. On répartit les tours
sur la durée typique du tournoi, ce qui garantit l'ordre chronologique : tous les matchs d'un tour ont lieu avant
ceux du tour suivant, et les poules (`RR`) avant la phase finale.

Durées retenues : Grand Chelem 14 jours, Masters à 96 joueurs et plus 12 jours, JO 8 jours, Coupe Davis 3 jours,
compétitions par équipes avec poules 10 jours, le reste 7 jours.
"""

from __future__ import annotations

from collections.abc import Sequence

ROUND_ROBIN = "RR"
ELIMINATION_ORDER: dict[str, int] = {"R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6, "BR": 6}


def tournament_duration_days(level: str, draw_size: int, has_round_robin: bool) -> int:
    """Durée typique d'un tournoi, en jours."""
    if level == "G":
        return 14
    if level == "M":
        return 12 if draw_size >= 96 else 7
    if level == "O":
        return 8
    if level == "D":
        return 3
    if level == "A" and has_round_robin:
        return 10
    return 7


def _spread(count: int, span: int) -> list[int]:
    """Répartit `count` éléments ordonnés sur `span` jours (0 … span-1), sans jamais revenir en arrière."""
    if count <= 1:
        return [0] * count
    return [(index * span) // count for index in range(count)]


def estimate_day_offsets(
    rounds: Sequence[str],
    match_nums: Sequence[int],
    level: str,
    draw_size: int,
) -> list[int]:
    """Jour de chaque match (0 = jour de `tourney_date`), dans l'ordre des listes reçues.

    Args:
        rounds: tour de chaque match (`R32`, `QF`, `RR`…).
        match_nums: numéro du match dans le tournoi (ordonne les matchs de poule).
        level: niveau du tournoi (`G`, `M`, `A`, `D`, `F`, `O`).
        draw_size: taille du tableau.

    Raises:
        ValueError: si les listes n'ont pas la même longueur ou si un tour est inconnu.
    """
    if len(rounds) != len(match_nums):
        msg = "rounds et match_nums doivent avoir la même longueur"
        raise ValueError(msg)
    unknown = {r for r in rounds if r != ROUND_ROBIN and r not in ELIMINATION_ORDER}
    if unknown:
        msg = f"tour(s) inconnu(s) : {sorted(unknown)}"
        raise ValueError(msg)

    has_rr = ROUND_ROBIN in rounds
    duration = tournament_duration_days(level, draw_size, has_rr)
    elimination_levels = sorted({ELIMINATION_ORDER[r] for r in rounds if r != ROUND_ROBIN})

    rr_span = duration - len(elimination_levels) if elimination_levels else duration
    rr_span = max(rr_span, 1) if has_rr else 0
    elimination_span = duration - rr_span

    elimination_day: dict[int, int] = {}
    if elimination_levels:
        count = len(elimination_levels)
        last = max(elimination_span - 1, count - 1)  # jamais deux tours le même jour
        for index, level_order in enumerate(elimination_levels):
            day = rr_span + (round(index * last / (count - 1)) if count > 1 else last)
            elimination_day[level_order] = day

    rr_positions = sorted(
        (num, pos) for pos, (rnd, num) in enumerate(zip(rounds, match_nums, strict=True)) if rnd == ROUND_ROBIN
    )
    rr_days = dict(zip((pos for _, pos in rr_positions), _spread(len(rr_positions), rr_span), strict=True))

    return [
        rr_days[pos] if rnd == ROUND_ROBIN else elimination_day[ELIMINATION_ORDER[rnd]]
        for pos, rnd in enumerate(rounds)
    ]
