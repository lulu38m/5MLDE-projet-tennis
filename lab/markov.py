"""Probabilité de gagner un match à partir des points gagnés au service, par chaîne de Markov.

Modèle classique (Barnett et Clarke, O'Malley) : chaque point est indépendant, le serveur gagne son point avec une
probabilité fixe. On en déduit le jeu, le tie-break, le set puis le match (au meilleur des 3 ou des 5 sets).
"""

from __future__ import annotations

from functools import cache, lru_cache


def game(p: float) -> float:
    """Le serveur gagne son jeu (point gagné avec la probabilité `p`)."""
    q = 1 - p
    deuce = p * p / (p * p + q * q) if p * p + q * q > 0 else 0.5
    return p**4 * (1 + 4 * q + 10 * q * q) + 20 * p**3 * q**3 * deuce


def tiebreak(pa: float, pb: float) -> float:
    """A gagne le tie-break en servant le premier point (`pa`, `pb` : points gagnés au service de A et de B)."""

    @cache
    def win(a: int, b: int) -> float:
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        if a == b and a >= 6:  # égalité : deux points, un sur chaque service
            won, lost = pa * (1 - pb), (1 - pa) * pb
            return won / (won + lost) if won + lost > 0 else 0.5
        served = a + b
        a_serves = served % 4 in (0, 3)  # A, B, B, A, A, B, B…
        p = pa if a_serves else 1 - pb
        return p * win(a + 1, b) + (1 - p) * win(a, b + 1)

    return win(0, 0)


def set_(pa: float, pb: float, a_serves_first: bool) -> float:
    """A gagne le set (jeu décisif à 6-6)."""
    ga, gb, tb = game(pa), game(pb), tiebreak(pa, pb) if a_serves_first else 1 - tiebreak(pb, pa)

    @cache
    def win(a: int, b: int) -> float:
        if (a >= 6 and a - b >= 2) or a == 7:
            return 1.0
        if (b >= 6 and b - a >= 2) or b == 7:
            return 0.0
        if a == 6 and b == 6:
            return tb
        a_serves = ((a + b) % 2 == 0) == a_serves_first
        p = ga if a_serves else 1 - gb
        return p * win(a + 1, b) + (1 - p) * win(a, b + 1)

    return win(0, 0)


@lru_cache(maxsize=200_000)
def _match(pa: float, pb: float, best_of: int) -> float:
    s = 0.5 * (set_(pa, pb, True) + set_(pa, pb, False))  # qui sert en premier : tiré au sort
    q = 1 - s
    if best_of == 5:
        return s**3 * (1 + 3 * q + 6 * q * q)
    return s * s * (1 + 2 * q)


def match(pa: float, pb: float, best_of: int = 3) -> float:
    """A gagne le match ; `pa`, `pb` arrondis au millième (cache)."""
    return _match(_clip(pa), _clip(pb), 5 if best_of == 5 else 3)


def _clip(x: float) -> float:
    return min(max(round(x, 3), 0.30), 0.90)
