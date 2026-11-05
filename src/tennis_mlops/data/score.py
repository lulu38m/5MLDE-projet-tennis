"""Interprétation de la colonne `score` des fichiers Sackmann.

Le profilage 2010-2026 montre des formats variés : sets classiques (`6-4 7-6(5)`), super tie-break
(`6-3 3-6 [10-8]`), abandons (`RET`), forfaits (`W/O`, `Walkover`), disqualifications (`DEF`, `Def.`,
`Default`) et matchs interrompus (`Played and abandoned`, `ABD`). Cette classification sert au statut renvoyé par
`mock_source` et aux contrôles qualité.
"""

from __future__ import annotations

import re
from enum import StrEnum

_SET = r"\d{1,2}-\d{1,2}(?:\(\d+\))?"
_MATCH_TIEBREAK = r"\[\d{1,2}-\d{1,2}\]"
_COMPLETED = re.compile(rf"^{_SET}(?: {_SET})*(?: {_MATCH_TIEBREAK})?$")


class MatchStatus(StrEnum):
    """Issue d'un match, déduite du score."""

    COMPLETED = "completed"
    RETIRED = "retired"
    WALKOVER = "walkover"
    DEFAULTED = "defaulted"
    ABANDONED = "abandoned"
    UNKNOWN = "unknown"


def classify_score(score: str | None) -> MatchStatus:
    """Déduit l'issue d'un match à partir de son score brut."""
    if score is None or not score.strip():
        return MatchStatus.UNKNOWN
    text = score.strip()
    lowered = text.lower()
    if "w/o" in lowered or "walkover" in lowered:
        return MatchStatus.WALKOVER
    if "abandoned" in lowered or "abd" in lowered:
        return MatchStatus.ABANDONED
    if re.search(r"\bRET\b", text):
        return MatchStatus.RETIRED
    if re.search(r"\bdef(?:ault)?\b", lowered):
        return MatchStatus.DEFAULTED
    if _COMPLETED.match(text):
        return MatchStatus.COMPLETED
    return MatchStatus.UNKNOWN
