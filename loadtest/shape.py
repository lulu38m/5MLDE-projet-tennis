"""Profil de charge de la campagne : les paliers de `LOAD_PROFILE` (voir `profiles.py`).

Fichier à part de `locustfile.py` : chargé seulement par la campagne (`-f loadtest/locustfile.py,loadtest/shape.py`).
Sans lui, l'interface web de Locust (`make load-ui`) laisse choisir le nombre d'utilisateurs à la main.
"""

from __future__ import annotations

import os

from locust import LoadTestShape

from loadtest.profiles import from_env, tick


class ProfileShape(LoadTestShape):
    """Suit les paliers, puis arrête le test."""

    def __init__(self) -> None:
        super().__init__()
        self.profile = from_env(os.environ.get("LOAD_PROFILE", "[]"))

    def tick(self) -> tuple[int, float] | None:
        """(utilisateurs, vitesse d'arrivée), ou None pour arrêter."""
        return tick(self.profile, self.get_run_time())
