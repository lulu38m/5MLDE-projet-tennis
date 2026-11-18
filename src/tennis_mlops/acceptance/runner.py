"""Déroulé de `make scenarios` : vérifications préalables, scénarios dans l'ordre, remise en ordre, rapport.

1. **La stack répond** (source, API, MLflow, Prefect, alertes, interface) et part de l'**état de référence** de
   `make up` : une seule version, en production, horloge au début du rejeu. Les scénarios s'enchaînent sur ce
   point de départ connu ; leurs résultats sont donc les mêmes sur toutes les machines. Sinon : `make reset &&
   make up` (`--force` pour jouer quand même, sans garantie sur S1).
2. **Pas de carte Teams** pendant les tests : le service d'alertes passe en mode silence (les alertes sont reçues
   et gardées, statut `silenced`) ; `TEAMS=1` les laisse partir. La simulation planifiée est mise en pause.
3. **Chaque scénario** est joué même si le précédent a échoué ; entre deux, l'incident est arrêté et les
   entraînements déclenchés par l'ingestion vont à leur terme.
4. **À la fin**, quoi qu'il arrive : incident arrêté, services relancés, silence levé, rapport écrit.
"""

from __future__ import annotations

import platform
import subprocess
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tennis_mlops.acceptance.report import CampaignResult, Recorder, ScenarioFailedError, ScenarioResult, Status
from tennis_mlops.acceptance.scenarios import SCENARIOS, Actions, Context, Scenario, settle
from tennis_mlops.acceptance.services import ServiceControl, ServiceControlError
from tennis_mlops.acceptance.stack import Stack, WaitTimeoutError, now
from tennis_mlops.registry.tagging import Alias

SERVICES = ("mock_source", "api", "mlflow", "prefect", "alert_sink", "ui")
STOPPABLE = ("mock_source", "mlflow")  # services que S4 et S8 arrêtent : relancés à la fin s'ils sont à terre
SILENCE_MIN = 90
REPORT_DIR = Path("state") / "acceptance"


class NotReadyError(RuntimeError):
    """La stack ne répond pas, ou n'est pas dans l'état de référence."""


@dataclass
class Campaign:
    """Une passe de tests d'acceptation."""

    stack: Stack
    services: ServiceControl
    actions: Actions
    scenarios: Sequence[Callable[[], Scenario]] = SCENARIOS
    only: frozenset[str] = frozenset()
    teams: bool = False
    force: bool = False
    log: Callable[[str], None] = print
    notes: list[str] = field(default_factory=list)
    since: datetime = field(default_factory=now)  # début de la campagne : les entraînements suivis partent d'ici

    # ------------------------------------------------------------------ vérifications préalables

    def preflight(self) -> None:
        """La stack répond et part de l'état de référence ; sinon `NotReadyError` avec la commande à lancer."""
        stack = self.stack
        down = [name for name in SERVICES if not stack.healthy(name)]
        if down:
            msg = f"services injoignables : {', '.join(down)} — la stack tourne-t-elle ? (make up, puis make ps)"
            raise NotReadyError(msg)
        production, versions = stack.alias(Alias.PRODUCTION), stack.version_count()
        clock, start = stack.clock(), stack.config.simulation.replay_start
        problems = []
        if production is None:
            problems.append("aucune version en production")
        if versions != 1:
            problems.append(f"{versions} versions dans le registre (attendu : 1, celle de make up)")
        if clock != start:
            problems.append(f"horloge simulée au {clock} (attendu : {start}, début du rejeu)")
        if problems and not self.force:
            msg = ("la stack n'est pas dans l'état laissé par `make up` : " + " ; ".join(problems) + ". Repartir "
                   "d'une stack neuve : make reset && make up (ou FORCE=1 pour jouer quand même)")  # fmt: skip
            raise NotReadyError(msg)
        if problems:
            self.notes.append("joué hors de l'état de référence (FORCE=1) : " + " ; ".join(problems))
        self.log(f"état de départ : production v{production}, {versions} version(s), horloge au {clock}")

    # ------------------------------------------------------------------ déroulé

    def run(self) -> CampaignResult:
        """Tous les scénarios demandés, puis la remise en ordre ; ne lève pas pour un scénario raté."""
        started = self.since = now()
        environment = self.environment()
        self.preflight()
        self._quiet()
        results: list[ScenarioResult] = []
        try:
            for factory in self.scenarios:
                scenario = factory()
                if self.only and scenario.id not in self.only:
                    results.append(ScenarioResult(scenario.id, scenario.title, scenario.expected, Status.SKIPPED))
                    continue
                results.append(self.play(scenario))
        finally:
            self.cleanup()
        return CampaignResult(started, now(), environment, results, self.notes)

    def play(self, scenario: Scenario) -> ScenarioResult:
        """Un scénario, ses vérifications et sa durée."""
        self.log(f"\n▶ {scenario.id} · {scenario.title}")
        rec = Recorder()
        ctx = Context(self.stack, self.services, self.actions, rec)
        status, error = Status.PASSED, ""
        began = time.monotonic()

        try:
            scenario.run(ctx)
        except ScenarioFailedError:
            status = Status.FAILED
        except (WaitTimeoutError, ServiceControlError, NotReadyError) as exc:
            status, error = Status.ERROR, f"{type(exc).__name__} : {exc}"
        except Exception as exc:
            status, error = Status.ERROR, "".join(traceback.format_exception(exc, limit=6))
        finally:
            for step in rec.steps:
                self.log(f"  {'✅' if step.ok else '❌'} {step.text}")
            if error:
                self.log(f"  ⚠️  {error.strip().splitlines()[-1]}")
            try:
                self.between()
            except Exception as exc:  # la remise en ordre ne doit pas masquer le résultat du scénario
                self.log(f"  ⚠️  remise en ordre incomplète : {type(exc).__name__} : {exc}")
        seconds = time.monotonic() - began
        result = ScenarioResult(scenario.id, scenario.title, scenario.expected, status, seconds, rec.steps,
                                rec.links, error)  # fmt: skip
        self.log(f"  → {result.status.value} en {seconds:.0f} s")
        return result

    def between(self) -> None:
        """Entre deux scénarios : plus d'incident, services relancés, entraînements en cours terminés."""
        stack = self.stack
        self._restart_stopped()
        stack.chaos("off")
        settle(stack, self.since)

    def cleanup(self) -> None:
        """À la fin, même après une erreur : incident arrêté, services relancés, silence levé."""
        stack = self.stack
        try:
            self._restart_stopped()
            stack.chaos("off")
        except Exception as exc:
            self.log(f"⚠️  remise en ordre : {type(exc).__name__} : {exc}")
        if not self.teams:
            try:
                stack.unsilence()
            except Exception as exc:
                self.log(f"⚠️  mode silence non levé ({type(exc).__name__}) : il s'arrêtera seul dans {SILENCE_MIN} min")

    # ------------------------------------------------------------------ outils

    def _quiet(self) -> None:
        stack = self.stack
        if stack.pause_simulation():
            self.notes.append("simulation planifiée mise en pause (make demo pour la relancer)")
        teams = bool(stack.alert_health().get("teams"))
        if self.teams:
            self.notes.append("TEAMS=1 : les cartes partent sur Teams" if teams else "aucun webhook Teams configuré")
            return
        stack.silence(SILENCE_MIN, "tests d'acceptation (make scenarios)")
        if teams:
            self.notes.append("mode silence : les alertes sont reçues et gardées, aucune carte n'est envoyée sur Teams")

    def _restart_stopped(self) -> None:
        for service in STOPPABLE:
            if not self.stack.healthy(service):
                self.log(f"  ↻ {service} relancé")
                self.services.start(service)
                self.stack.wait_healthy(service, timeout_s=240)
                if service == "mock_source":
                    self.stack.set_clock(self.stack.watermark())

    def environment(self) -> dict[str, str]:
        """Ce qui situe la campagne : machine, révision du code, adresses."""
        return {
            "machine": f"{platform.node()} · {platform.system()} {platform.machine()}",
            "Python": platform.python_version(),
            "révision": _git_revision(self.stack.root),
            "API": self.stack.url("api"),
            "Prefect": self.stack.url("prefect"),
            "MLflow": str(self.stack.settings.mlflow_ui_url).rstrip("/"),
        }


def _git_revision(root: Path) -> str:
    try:
        done = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=root, capture_output=True,
                              text=True, timeout=10, check=False)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return "?"
    return done.stdout.strip() or "?"
