"""Incidents contrôlés, pour démontrer retries, alertes et contrôles qualité bloquants.

| Mode | Effet sur les endpoints de données (jamais sur `/admin/*` ni `/health`) |
|---|---|
| `off` | Fonctionnement normal |
| `503` | Réponse `503 Service Unavailable` + `Retry-After` |
| `slow` | Réponse retardée de `delay_s` secondes (pour déclencher les timeouts) |
| `corrupt` | ~60 % des matchs altérés, toujours les mêmes (voir `corrupt_payload`) |
| `schema` | Contrat rompu : `match_id` renommé, `score` supprimé, `minutes` en texte, `winner.id` renommé |

`failures` limite le nombre de requêtes affectées (puis retour automatique à `off`) : `{"mode": "503",
"failures": 2}` fait échouer deux appels puis laisse passer le troisième, ce qui montre un retry réussi.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

RECORD_KEYS = ("results", "fixtures")


class ChaosMode(StrEnum):
    """Type d'incident simulé."""

    OFF = "off"
    UNAVAILABLE = "503"
    SLOW = "slow"
    CORRUPT = "corrupt"
    SCHEMA = "schema"


class ChaosSettings(BaseModel):
    """Corps de `POST /admin/chaos` et état renvoyé par `GET /admin/chaos`."""

    model_config = ConfigDict(extra="forbid")

    mode: ChaosMode = ChaosMode.OFF
    failures: int | None = Field(default=None, ge=1, description="Nombre de requêtes affectées ; null = illimité")
    delay_s: float = Field(default=5.0, ge=0, le=120)


class ChaosController:
    """État partagé du chaos (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settings = ChaosSettings()

    def configure(self, settings: ChaosSettings) -> ChaosSettings:
        """Active un mode (ou `off`)."""
        with self._lock:
            self._settings = settings.model_copy()
            return self._settings.model_copy()

    def state(self) -> ChaosSettings:
        """Mode courant."""
        with self._lock:
            return self._settings.model_copy()

    def consume(self) -> ChaosSettings:
        """Réglage à appliquer à la requête de données en cours ; décrémente le compteur `failures`."""
        with self._lock:
            current = self._settings.model_copy()
            if current.mode is not ChaosMode.OFF and self._settings.failures is not None:
                remaining = self._settings.failures - 1
                self._settings = (
                    ChaosSettings() if remaining == 0 else self._settings.model_copy(update={"failures": remaining})
                )
            return current


def _bucket(match_id: str) -> int:
    return int(hashlib.sha256(match_id.encode("utf-8")).hexdigest()[:8], 16) % 10


def corrupt_payload(payload: dict[str, Any], now: date) -> dict[str, Any]:
    """Altère environ 60 % des matchs d'une réponse, toujours les mêmes (déterministe)."""
    corrupted = dict(payload)
    for key in RECORD_KEYS:
        if key not in payload:
            continue
        records: list[dict[str, Any]] = []
        for record in payload[key]:
            bucket = _bucket(str(record["match_id"]))
            altered = {k: dict(v) if isinstance(v, dict) else v for k, v in record.items()}
            if bucket == 0 and "winner" in altered:
                altered["loser"]["id"] = altered["winner"]["id"]
            elif bucket == 1 and "score" in altered:
                altered["score"] = "6-4 7-"
            elif bucket == 2 and "minutes" in altered:
                altered["minutes"] = -12
            elif bucket == 3:
                altered["match_date"] = (now + timedelta(days=30)).isoformat()
            elif bucket == 4:
                altered["surface"] = "Moon"
            elif bucket == 5:
                continue  # ligne perdue : le volume ne correspond plus à expected_matches
            records.append(altered)
        corrupted[key] = records
        if "count" in corrupted:
            corrupted["count"] = len(records)
    return corrupted


def break_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Rompt le contrat JSON (renommage, suppression, changement de type)."""
    broken = dict(payload)
    for key in RECORD_KEYS:
        if key not in payload:
            continue
        records = []
        for record in payload[key]:
            altered = {("matchId" if k == "match_id" else k): v for k, v in record.items() if k != "score"}
            if "minutes" in altered:
                altered["minutes"] = f"{altered['minutes']} min"
            for side in ("winner", "player1"):
                if isinstance(altered.get(side), dict):
                    player = dict(altered[side])
                    player["player_id"] = player.pop("id")
                    altered[side] = player
            records.append(altered)
        broken[key] = records
    return broken
