"""Scénarios Locust contre l'API de prédiction (phase 10).

Trois types d'utilisateurs, chacun envoyant environ une requête par seconde (attente de 0,5 à 1,5 s) :

- `PredictUser` : `POST /predict`, un match entre deux joueurs tirés au hasard ;
- `BatchUser` : `POST /predict/batch` de 50 matchs (un tableau de tournoi à prédire d'un coup) ;
- `MixUser` : un usage réaliste de l'interface et des clients : surtout des prédictions, des recherches de
  joueurs, des fiches joueur, des face-à-face, parfois un batch ou l'état du modèle.

Les joueurs sont lus une fois par processus (`GET /players`, hors statistiques). Chaque requête porte
`X-Client: loadtest` : elle est tracée et comptée à part dans PostgreSQL et Prometheus.

Si `LOAD_LOG_DIR` est défini, chaque requête est aussi écrite dans `requests-<pid>.csv` (heure, endpoint, durée,
succès) : la campagne en tire les percentiles exacts de chaque palier (`analysis.py`).
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

from locust import FastHttpUser, between, events, task
from locust.env import Environment

from loadtest.common import BATCH_SIZE, CLIENT, TIMEOUT_S, RequestLog, match, players


class _ApiUser(FastHttpUser):
    abstract = True
    wait_time = between(0.5, 1.5)
    network_timeout = TIMEOUT_S
    connection_timeout = TIMEOUT_S

    def on_start(self) -> None:
        """Les joueurs du processus (lus au premier utilisateur)."""
        self.pool = players(self.host or "")

    def predict(self) -> None:
        """Une prédiction."""
        self.client.post("/predict", json=match(self.pool), headers=CLIENT)

    def batch(self) -> None:
        """Un batch de 50 matchs."""
        body = {"matches": [match(self.pool) for _ in range(BATCH_SIZE)]}
        self.client.post("/predict/batch", json=body, headers=CLIENT, name=f"/predict/batch ({BATCH_SIZE})")


class PredictUser(_ApiUser):
    """Une prédiction à la fois."""

    @task
    def one(self) -> None:
        """Une prédiction."""
        self.predict()


class BatchUser(_ApiUser):
    """Un batch de 50 matchs à la fois."""

    @task
    def fifty(self) -> None:
        """Un batch."""
        self.batch()


class MixUser(_ApiUser):
    """Usage réaliste : 60 % de prédictions, le reste en consultation, un batch de temps en temps."""

    @task(12)
    def one(self) -> None:
        """Une prédiction."""
        self.predict()

    @task(3)
    def search(self) -> None:
        """Une recherche de joueur par un morceau de nom."""
        name = str(random.choice(self.pool).get("name") or "")
        word = max(name.split(), key=len, default="a")[:5]
        self.client.get("/players", params={"search": word, "limit": 10}, headers=CLIENT, name="/players?search")

    @task(2)
    def stats(self) -> None:
        """La fiche d'un joueur."""
        player = random.choice(self.pool)["id"]
        self.client.get(f"/players/{player}/stats", headers=CLIENT, name="/players/[id]/stats")

    @task(2)
    def h2h(self) -> None:
        """Un face-à-face."""
        p1, p2 = random.sample(self.pool, 2)
        self.client.get("/h2h", params={"p1": p1["id"], "p2": p2["id"]}, headers=CLIENT, name="/h2h")

    @task(1)
    def fifty(self) -> None:
        """Un batch, de temps en temps."""
        self.batch()


_log: RequestLog | None = None


@events.init.add_listener
def _open_log(environment: Environment, **_: Any) -> None:  # noqa: ANN401
    global _log
    directory = os.environ.get("LOAD_LOG_DIR")
    if directory and _log is None:
        _log = RequestLog(Path(directory))


@events.request.add_listener
def _log_request(name: str, response_time: float, exception: BaseException | None, **_: Any) -> None:  # noqa: ANN401
    if _log is not None:
        _log.add(name, response_time, exception is None)


@events.test_start.add_listener
def _mark_start(environment: Environment, **_: Any) -> None:  # noqa: ANN401
    directory = os.environ.get("LOAD_LOG_DIR")
    if directory:
        (Path(directory) / f"start-{os.getpid()}.txt").write_text(f"{time.time():.3f}\n", encoding="utf-8")


@events.quitting.add_listener
def _close_log(environment: Environment, **_: Any) -> None:  # noqa: ANN401
    if _log is not None:
        _log.close()
