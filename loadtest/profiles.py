"""Les scénarios d'une campagne (`configs/load.yaml`) et leur profil de charge : une suite de paliers.

Un palier = une durée, un nombre d'utilisateurs visé, une vitesse d'arrivée. Locust les suit (`shape.py`, profil
passé en JSON par la variable `LOAD_PROFILE`) ; l'analyse mesure chaque palier en écartant sa montée
(`warmup_s`), le temps que les utilisateurs arrivent et que les latences se stabilisent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_LOAD_CONFIG = Path("configs/load.yaml")
UserKind = Literal["mix", "predict", "batch"]
USER_CLASSES: dict[str, str] = {"mix": "MixUser", "predict": "PredictUser", "batch": "BatchUser"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Stage(_Strict):
    """Un palier : `users` utilisateurs pendant `duration_s`, arrivés à `spawn_rate` par seconde."""

    duration_s: float = Field(gt=0)
    users: int = Field(ge=0)
    spawn_rate: float = Field(gt=0)
    measured_from_s: float = Field(default=0, ge=0, description="Début de la mesure, depuis le début du palier")
    label: str = ""


class Scenario(_Strict):
    """Un scénario de `configs/load.yaml`."""

    name: str
    config: Literal["A", "B", "C"]
    users: UserKind = "mix"
    # rampe par paliers
    steps: list[int] | None = None
    step_s: float = 120
    warmup_s: float = 20
    # pic
    base: int | None = None
    factor: float = 5
    ramp_s: float = 10
    before_s: float = 60
    during_s: float = 90
    after_s: float = 120
    # endurance
    constant: int | None = None
    duration_s: float = 900

    @property
    def kind(self) -> Literal["ramp", "spike", "endurance"]:
        """Le type de scénario, déduit de ses paramètres."""
        if self.steps:
            return "ramp"
        if self.base is not None:
            return "spike"
        if self.constant is not None:
            return "endurance"
        raise ValueError(f"scénario {self.name} : ni steps, ni base, ni constant")

    @property
    def slug(self) -> str:
        """Nom de dossier : `B-ramp`, `B-spike`, etc."""
        return f"{self.config}-{self.name}"

    def stages(self) -> list[Stage]:
        """Le profil de charge du scénario."""
        if self.kind == "ramp":
            return ramp(self.steps or [], self.step_s, self.warmup_s)
        if self.kind == "spike":
            return spike(self.base or 1, self.factor, self.ramp_s, self.before_s, self.during_s, self.after_s)
        users = self.constant or 1
        return [Stage(duration_s=self.duration_s, users=users, spawn_rate=max(users / 10, 1),
                      measured_from_s=self.warmup_s, label=f"{users} utilisateurs")]  # fmt: skip

    def duration(self) -> float:
        """Durée totale (s)."""
        return sum(s.duration_s for s in self.stages())


class Slo(_Strict):
    """Ce qu'un palier doit tenir."""

    p95_ms: float = 200
    batch_p95_ms: float = 1000
    error_rate: float = 0.01


class LocustSettings(_Strict):
    """Réglages de Locust."""

    processes: int = Field(default=4, ge=1)
    timeout_s: float = Field(default=10, gt=0)
    players: int = Field(default=200, ge=2, le=200)


class LoadConfig(_Strict):
    """`configs/load.yaml`."""

    slo: Slo = Slo()
    locust: LocustSettings = LocustSettings()
    configs: dict[str, str]
    campaigns: dict[str, list[Scenario]]

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_LOAD_CONFIG) -> LoadConfig:
        """Lit le fichier."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def ramp(steps: list[int], step_s: float, warmup_s: float) -> list[Stage]:
    """Rampe par paliers : chaque palier est atteint en `warmup_s / 2` au plus, puis tenu."""
    out, previous = [], 0
    for users in steps:
        rate = max(abs(users - previous) / max(warmup_s / 2, 1), 1)
        out.append(Stage(duration_s=step_s, users=users, spawn_rate=rate, measured_from_s=warmup_s,
                         label=f"{users} utilisateurs"))  # fmt: skip
        previous = users
    return out


def spike(base: int, factor: float, ramp_s: float, before_s: float, during_s: float, after_s: float) -> list[Stage]:
    """Pic : `base` utilisateurs, puis `base * factor` arrivés en `ramp_s`, puis retour à `base`."""
    peak = round(base * factor)
    rate = max((peak - base) / ramp_s, 1)
    return [
        Stage(duration_s=before_s, users=base, spawn_rate=max(base / 5, 1), measured_from_s=min(10, before_s / 2),
              label="avant"),
        Stage(duration_s=during_s, users=peak, spawn_rate=rate, label="pic"),
        Stage(duration_s=after_s, users=base, spawn_rate=rate, label="après"),
    ]  # fmt: skip


def to_env(stages: list[Stage]) -> str:
    """Le profil pour `LOAD_PROFILE` (lu par `shape.py`)."""
    return json.dumps([[s.duration_s, s.users, s.spawn_rate] for s in stages])


def from_env(value: str) -> list[tuple[float, int, float]]:
    """L'inverse de `to_env`."""
    return [(float(d), int(u), float(r)) for d, u, r in json.loads(value)]


def tick(profile: list[tuple[float, int, float]], run_time: float) -> tuple[int, float] | None:
    """(utilisateurs, vitesse) à l'instant `run_time` du test ; None une fois le profil terminé."""
    elapsed = 0.0
    for duration, users, rate in profile:
        elapsed += duration
        if run_time < elapsed:
            return users, rate
    return None
