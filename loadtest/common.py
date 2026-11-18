"""Ce que les scénarios Locust partagent, sans importer Locust.

Locust patche le réseau avec gevent dès son import : ce module-ci (joueurs, fabrication d'un match, journal des
requêtes) reste importable par les tests (`tests/loadtest/`).
"""

from __future__ import annotations

import csv
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, TextIO

import requests

CLIENT = {"X-Client": "loadtest"}
SURFACES = ("Hard", "Hard", "Hard", "Clay", "Clay", "Grass")  # à peu près la part de chaque surface sur le circuit
BATCH_SIZE = 50
TIMEOUT_S = float(os.environ.get("LOAD_TIMEOUT_S", "10"))
PLAYERS = int(os.environ.get("LOAD_PLAYERS", "200"))

_players: list[dict[str, Any]] = []
_lock = threading.Lock()


def players(host: str) -> list[dict[str, Any]]:
    """Les joueurs classés de l'année (une seule lecture par processus)."""
    with _lock:
        if not _players:
            response = requests.get(f"{host}/players", params={"limit": PLAYERS}, headers=CLIENT, timeout=30)
            response.raise_for_status()
            found = response.json()["players"]
            if len(found) < 2:
                raise RuntimeError("moins de deux joueurs dans GET /players : la base est-elle ingérée ?")
            _players.extend(found)
    return _players


def match(pool: list[dict[str, Any]]) -> dict[str, Any]:
    """Un match entre deux joueurs différents, sur une surface tirée au hasard."""
    p1, p2 = random.sample(pool, 2)
    return {"surface": random.choice(SURFACES), "player1": {"id": p1["id"]}, "player2": {"id": p2["id"]}}


# ---------------------------------------------------------------------------- journal des requêtes (campagne)


class RequestLog:
    """Une ligne par requête, écrite par paquets dans `requests-<pid>.csv`."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = (directory / f"requests-{os.getpid()}.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["t", "name", "ms", "ok"])
        self._rows: list[tuple[float, str, float, int]] = []

    def add(self, name: str, response_time: float, ok: bool) -> None:
        """Une requête de plus."""
        self._rows.append((round(time.time(), 3), name, round(response_time, 2), int(ok)))
        if len(self._rows) >= 500:
            self.flush()

    def flush(self) -> None:
        """Écrit les lignes en attente."""
        self._writer.writerows(self._rows)
        self._rows.clear()
        self._file.flush()

    def close(self) -> None:
        """Dernière écriture."""
        self.flush()
        self._file.close()
