"""Les neuf scénarios d'acceptation (`make scenarios`) : chacun rejoue une situation et vérifie le résultat attendu.

| # | Scénario | Vérifié automatiquement |
|---|---|---|
| S1 | Nominal | lot ingéré → GX OK → nouvelle version → alias `production` déplacé → servie par l'API → feedbacks |
| S2 | Lot corrompu (`corrupt`) | expectation custom en échec, lot en quarantaine, aucun entraînement, alerte reçue |
| S3 | Source en `503` / `slow` | retries visibles dans Prefect, succès final ; alerte « dégradé » après épuisement |
| S4 | Source arrêtée | run « Dégradé », watermark inchangé, alerte ; l'API sert toujours le dernier état validé |
| S5 | Schéma modifié (`schema`) | contrôle de schéma bloquant : quarantaine, rien de stocké, alerte |
| S6 | Modèle moins bon | `gate_status=failed`, `production` inchangé |
| S7 | Rollback | `production` = ancienne version, l'API la sert sans redémarrage |
| S8 | MLflow arrêté + entrée invalide | modèle servi depuis le cache, `/ready` dégradé ; `422` tracé dans `errors` |
| S9 | Interface de test | l'UI répond, une prédiction faite depuis l'UI est dans `predictions` avec le client `ui` |

Les scénarios s'enchaînent dans cet ordre sur la même stack (S7 annule la promotion faite par S1) et partent de
l'état laissé par `make up` : v1 en production, horloge au début du rejeu. Chaque incident est arrêté avant de
passer au suivant, et un service arrêté est toujours relancé, même si le scénario échoue.
"""

from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar, Protocol

from tennis_mlops.acceptance.report import Recorder
from tennis_mlops.acceptance.services import ServiceControl
from tennis_mlops.acceptance.stack import TRAINING_FLOW, RunInfo, Stack, WaitTimeoutError, now, wait_for
from tennis_mlops.config import Settings
from tennis_mlops.registry.tagging import Alias

# Djokovic contre Nadal en demi-finale de Masters sur terre battue (comme `make predict`).
MATCH: dict[str, Any] = {"surface": "Clay", "tourney_level": "M", "round": "SF", "best_of": 3,
                         "player1": {"id": 104925}, "player2": {"id": 104745}}  # fmt: skip
INVALID_MATCH: dict[str, Any] = {**MATCH, "surface": "Moon"}  # surface inconnue : 422
CUSTOM_EXPECTATIONS = frozenset(
    {"expect_tennis_score_to_be_valid", "expect_winner_and_loser_to_differ", "expect_no_match_after_simulated_now",
     "expect_batch_volume_to_match_draw"}
)  # fmt: skip
SCHEMA_EXPECTATION = "expect_column_to_exist"
DEGRADED = "Dégradé"
TRAINING_TIMEOUT_S = 900  # une campagne prend une à trois minutes sur un portable
AUTOMATION_GRACE_S = 120  # délai pour que l'automation crée le run d'entraînement demandé
API_REFRESH_TIMEOUT_S = 120  # l'API relit l'alias toutes les `api.model_refresh_s` (30 s)
DELIVERY = {"sent": "carte envoyée sur Teams", "disabled": "pas de webhook : carte gardée",
            "silenced": "mode silence : carte gardée", "filtered": "type filtré : carte gardée",
            "repeated": "déjà signalé : carte gardée", "failed": "envoi Teams en échec"}  # fmt: skip


# --------------------------------------------------------------------------- actions (flows lancés depuis l'hôte)


@dataclass(frozen=True)
class FlowOutcome:
    """L'issue d'un flow lancé par un scénario."""

    flow_run_id: str | None
    state_name: str  # Completed, Failed, Dégradé…
    failed: bool
    result: Any = None  # le rapport du flow, ou l'exception s'il a échoué


class Actions(Protocol):
    """Ce que les scénarios font faire à la chaîne : les mêmes commandes que `make ingest` ou `make simulate`."""

    def ingest(self) -> FlowOutcome:
        """`make ingest`."""
        ...

    def simulate(self, days: int) -> FlowOutcome:
        """`make simulate DAYS=…`."""
        ...

    def train(self) -> FlowOutcome:
        """`make train` : une campagne lancée à la main."""
        ...

    def train_weak(self) -> int:
        """`make train-weak` ; code de sortie."""
        ...

    def rollback(self, reason: str) -> int:
        """`make rollback REASON=…` ; code de sortie."""
        ...

    def ui_smoke(self, ui_url: str, settings: Settings) -> list[str]:
        """`make ui-smoke` : les étapes réussies (lève `AssertionError` sinon)."""
        ...


class LocalActions:
    """Les flows et commandes du projet, lancés dans ce processus (comme `make` le ferait depuis l'hôte)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def ingest(self) -> FlowOutcome:
        """Un passage d'ingestion, suivi dans Prefect."""
        from tennis_mlops.flows.ingest import ingest_flow

        return _outcome(ingest_flow(return_state=True))

    def simulate(self, days: int) -> FlowOutcome:
        """Un pas de simulation : horloge, ingestion, feedback, scoring."""
        from tennis_mlops.flows.simulation import simulation_flow

        return _outcome(simulation_flow(days, return_state=True))

    def train(self) -> FlowOutcome:
        """Une campagne avec la configuration normale, suivie dans Prefect."""
        from tennis_mlops.flows.train import training_flow

        with _quiet():
            return _outcome(training_flow(return_state=True))

    def train_weak(self) -> int:
        """La campagne volontairement mauvaise."""
        from tennis_mlops.flows.train import main

        with _quiet():
            return main(["--config", str(self.root / "configs" / "train-weak.yaml")])

    def rollback(self, reason: str) -> int:
        """`production` ← `previous`."""
        from tennis_mlops.registry.__main__ import main

        with _quiet():
            return main(["rollback", "--reason", reason])

    def ui_smoke(self, ui_url: str, settings: Settings) -> list[str]:
        """Le test de fumée de l'interface."""
        import logging

        from tennis_mlops.ui.smoke import run

        # Streamlit reconfigure ses loggers à l'import : la page jouée hors serveur le signale à chaque exécution.
        logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
        return run(ui_url, settings)


@contextmanager
def _quiet() -> Iterator[None]:
    """Rien dans le terminal (bilans, barres de progression de MLflow) : les étapes du scénario disent l'essentiel."""
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


def _outcome(state: Any) -> FlowOutcome:  # noqa: ANN401
    run_id = state.state_details.flow_run_id
    return FlowOutcome(str(run_id) if run_id else None, str(state.name or state.type.value), bool(state.is_failed()),
                       state.result(raise_on_failure=False))  # fmt: skip


# --------------------------------------------------------------------------- contexte et base des scénarios


@dataclass
class Context:
    """Ce qu'un scénario a sous la main."""

    stack: Stack
    services: ServiceControl
    actions: Actions
    rec: Recorder


class Scenario(ABC):
    """Un scénario d'acceptation."""

    id: ClassVar[str]
    title: ClassVar[str]
    expected: ClassVar[str]

    @abstractmethod
    def run(self, ctx: Context) -> None:
        """Rejoue la situation et vérifie le résultat (`ctx.rec.expect`)."""


def settle(stack: Stack, since: datetime, timeout_s: float = TRAINING_TIMEOUT_S) -> list[RunInfo]:
    """Attend la fin des entraînements demandés depuis `since` ; renvoie leurs runs (du plus récent au plus ancien).

    L'ingestion ne lance pas l'entraînement : elle émet un événement auquel réagit l'automation Prefect, qui crée le
    run sur le worker de la stack. On attend donc d'abord que chaque demande ait son run, puis que tous finissent.
    """
    requested = len(stack.events("training_requested", since))
    started = now()

    def probe() -> list[RunInfo] | None:
        runs = stack.flow_runs([TRAINING_FLOW], since)
        if len(runs) < requested:
            if (now() - started).total_seconds() > AUTOMATION_GRACE_S:
                msg = (f"{requested} entraînement(s) demandé(s), {len(runs)} lancé(s) par l'automation après "
                       f"{AUTOMATION_GRACE_S} s : le worker tourne-t-il ? (make demo-status)")  # fmt: skip
                raise WaitTimeoutError(msg)
            return None
        return None if any(not r.final for r in runs) else runs

    return wait_for(probe, timeout_s, "fin des entraînements en cours", interval_s=5)


def _training_refused(ctx: Context) -> None:
    """Pendant la quarantaine, une campagne lancée à la main (`make train`) est interrompue avant d'entraîner."""
    latest = ctx.stack.latest_version()
    ctx.rec.step("entraînement lancé à la main pendant la quarantaine (make train)")
    training = ctx.actions.train()
    reason = str(training.result or "")
    ctx.rec.expect(training.failed and "quarantaine" in reason,
                   f"{training.state_name} : « {reason} »",
                   f"entraînement non interrompu : {training.state_name} {reason}")  # fmt: skip
    if training.flow_run_id:
        ctx.rec.link("Prefect : entraînement interrompu", ctx.stack.prefect_run_url(training.flow_run_id))
    ctx.rec.expect(ctx.stack.latest_version() == latest, f"aucune nouvelle version enregistrée (dernière : v{latest})")
    _alert(ctx, "echec", flow_run_id=training.flow_run_id)


def _quarantined(ctx: Context, since: datetime, flow_run_id: str | None) -> list[dict[str, Any]]:
    return [e for e in ctx.stack.events("lot_quarantined", since) if e.get("flow_run_id") == flow_run_id]


def _failed_expectations(events: list[dict[str, Any]]) -> set[str]:
    return {str(c.get("expectation")) for e in events for c in e["payload"].get("failed_checks", [])}


def _stored(ctx: Context, lot_ids: list[str]) -> int:
    return sum(int(ctx.stack.scalar("SELECT count(*) FROM matches WHERE lot_id = :l", l=lot)) for lot in lot_ids)


def _alert(ctx: Context, kind: str, flow_run_id: str | None = None, since: datetime | None = None) -> None:
    alert = ctx.stack.wait_alert(kind, flow_run_id=flow_run_id, since=since)
    delivery = DELIVERY.get(str(alert.get("delivery")), str(alert.get("delivery")))
    ctx.rec.expect(alert, f"alerte reçue par alert_sink : « {alert.get('title')} » ({delivery})")


def _probability(response: Any) -> str:  # noqa: ANN401
    try:
        body = response.json()
        return f"{float(body['player1']['win_probability']):.1%}"
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "?"


def _recover(ctx: Context, lot_ids: list[str]) -> None:
    """Incident arrêté : le même passage d'ingestion reprend les lots refusés (le watermark ne les avait pas passés)."""
    ctx.stack.chaos("off")
    again = ctx.actions.ingest()
    valid = {r["lot_id"] for r in ctx.stack.rows("SELECT lot_id FROM ingestion_lots WHERE status = 'valid'")}
    ctx.rec.expect(not again.failed and set(lot_ids) <= valid,
                   f"reprise : incident arrêté, les {len(lot_ids)} lot(s) refusé(s) passent ({again.state_name})",
                   f"reprise : {again.state_name}, lots toujours refusés : {sorted(set(lot_ids) - valid)}")  # fmt: skip


def _ingest_with(ctx: Context, mode: str, days: int, **chaos: Any) -> FlowOutcome:  # noqa: ANN401
    """Avance l'horloge, déclenche l'incident, ingère ; l'incident est toujours arrêté ensuite."""
    stack = ctx.stack
    step = stack.advance(days)
    ctx.rec.step(f"horloge {step.previous} → {step.now} : {step.new_matches} match(s), "
                 f"{len(step.tournaments)} tournoi(s) terminé(s)")  # fmt: skip
    stack.chaos(mode, **chaos)
    details = ", ".join(f"{k}={v}" for k, v in chaos.items())
    ctx.rec.step(f"incident « {mode} » activé sur la source" + (f" ({details})" if details else ""))
    try:
        outcome = ctx.actions.ingest()
    finally:
        stack.chaos("off")
    if outcome.flow_run_id:
        ctx.rec.link(f"Prefect : ingestion ({mode})", stack.prefect_run_url(outcome.flow_run_id))
    return outcome


# --------------------------------------------------------------------------- S1 → S9


class S1Nominal(Scenario):
    """La chaîne complète, sans incident."""

    id = "S1"
    title = "Nominal"
    expected = ("nouveau lot ingéré → GX OK → nouvelle version → alias production déplacé → l'API répond avec la "
                "nouvelle version → feedbacks enregistrés")  # fmt: skip
    # Jusqu'au 3 juin 2016 (Roland-Garros en cours) : cinq mois de données de plus que v1, assez pour que la nouvelle
    # version la batte vraiment. Mesuré : avec un, deux ou trois mois de plus, la règle la refuse à raison (log loss
    # légèrement moins bonne que v1, ou accuracy sous « le mieux classé gagne »).
    first_step_days = 154
    second_step_days = 7  # la semaine d'après : les matchs prédits au premier pas ont leur résultat

    def run(self, ctx: Context) -> None:
        """Deux pas de simulation, l'entraînement déclenché par événement entre les deux."""
        stack, rec = ctx.stack, ctx.rec
        before = stack.alias(Alias.PRODUCTION)
        rec.expect(before, f"production avant : v{before}", "aucune version en production : lancer `make up`")
        start = now()
        outcome = ctx.actions.simulate(self.first_step_days)
        report = outcome.result
        if outcome.flow_run_id:
            rec.link("Prefect : pas de simulation", stack.prefect_run_url(outcome.flow_run_id))
        rec.expect(not outcome.failed and getattr(report, "now", None),
                   f"pas de simulation : {getattr(report, 'previous', '?')} → {getattr(report, 'now', '?')}, "
                   f"{getattr(report, 'new_matches', 0)} match(s) terminé(s)",
                   f"pas de simulation : {outcome.state_name} — {getattr(report, 'detail', report)}")  # fmt: skip
        rec.expect(report.ingestion == "Completed" and report.ingestion_summary.startswith("ok"),
                   f"ingestion : {report.ingestion_summary}",
                   f"ingestion : {report.ingestion} — {report.ingestion_summary}")  # fmt: skip
        lots = stack.rows("SELECT lot_id, status FROM ingestion_lots WHERE updated_at >= :t", t=start)
        refused = [lot["lot_id"] for lot in lots if lot["status"] != "valid"]
        rec.expect(lots and not refused, f"Great Expectations : {len(lots)} lot(s) validé(s), aucun refusé",
                   f"Great Expectations : lots refusés {refused}")  # fmt: skip
        requests = stack.events("training_requested", start)
        asked = requests[0]["title"] if requests else ""
        rec.expect(requests, f"réentraînement demandé par l'ingestion : {asked}",
                   "aucun réentraînement demandé (moins de matchs que training_trigger_matches ?)")  # fmt: skip

        runs = settle(stack, start)
        done = [r for r in runs if r.state_type == "COMPLETED"]
        rec.expect(done, f"entraînement lancé par l'automation et terminé ({done[0].name if done else ''})",
                   f"entraînement : {[(r.name, r.state_name) for r in runs]}")  # fmt: skip
        rec.link("Prefect : entraînement", stack.prefect_run_url(done[0].id))
        new = stack.latest_version() or "?"
        tags = stack.version_tags(new) if new != "?" else {}
        rec.expect(new != before, f"nouvelle version enregistrée : v{new}", "aucune nouvelle version dans le registre")
        rec.link(f"MLflow : v{new}", stack.mlflow_version_url(new))
        reason = tags.get("gate_reason", "")
        rec.expect(tags.get("gate_status") == "passed", f"règle de mise en service passée : {reason[:160]}",
                   f"règle de mise en service : {tags.get('gate_status')} — {reason}")  # fmt: skip
        production, previous = stack.alias(Alias.PRODUCTION), stack.alias(Alias.PREVIOUS)
        rec.expect(production == new and previous == before,
                   f"alias production déplacé : v{before} → v{new} (previous = v{before})",
                   f"alias : production v{production}, previous v{previous}")  # fmt: skip

        wait_for(lambda: stack.served_version() == new, API_REFRESH_TIMEOUT_S, f"API sur v{new}")
        response = stack.predict(MATCH)
        answered = response.headers.get("x-model-version")
        rec.expect(response.status_code == 200 and answered == new,
                   f"l'API répond avec v{new}, sans redémarrage (Djokovic contre Nadal sur terre : "
                   f"{_probability(response)})",
                   f"l'API : HTTP {response.status_code}, version {answered}")  # fmt: skip
        scored = int(stack.scalar("SELECT count(*) FROM predictions WHERE client = 'scoring' AND created_at >= :t",
                                  t=start))  # fmt: skip
        rec.expect(scored > 0, f"scoring : {scored} match(s) à venir prédits par l'API",
                   "scoring : aucun match à venir prédit")  # fmt: skip

        second = now()
        follow = ctx.actions.simulate(self.second_step_days)
        result = follow.result
        rec.expect(not follow.failed, f"pas suivant : {getattr(result, 'previous', '?')} → "
                   f"{getattr(result, 'now', '?')} ; feedback : {getattr(result, 'feedback', '?')}",
                   f"pas suivant : {follow.state_name}")  # fmt: skip
        feedbacks = int(stack.scalar("SELECT count(*) FROM feedbacks WHERE updated_at >= :t", t=second))
        judged = stack.scalar("SELECT count(*) FROM prediction_outcomes WHERE correct IS NOT NULL")
        rec.expect(feedbacks > 0, f"feedbacks enregistrés : {feedbacks} vrai(s) résultat(s) remonté(s) à l'API, "
                   f"{judged} prédiction(s) jugée(s) au total", "aucun feedback enregistré")  # fmt: skip
        settle(stack, second)


class S2Corrupt(Scenario):
    """Un lot corrompu est bloqué par Great Expectations."""

    id = "S2"
    title = "Lot corrompu (corrupt)"
    expected = ("expectation custom en échec, lot en quarantaine, aucun entraînement (un entraînement lancé à la main "
                "est interrompu), alerte reçue par alert_sink")  # fmt: skip

    def run(self, ctx: Context) -> None:
        """Une semaine ingérée pendant que la source corrompt ses réponses."""
        stack, rec = ctx.stack, ctx.rec
        production = stack.alias(Alias.PRODUCTION)
        start = now()
        outcome = _ingest_with(ctx, "corrupt", 7)
        rec.expect(outcome.failed, f"ingestion en échec (état {outcome.state_name}) : `make ingest` sort en code 1",
                   f"ingestion : {outcome.state_name}, pas en échec")  # fmt: skip
        events = _quarantined(ctx, start, outcome.flow_run_id)
        lots = [str(e["lot_id"]) for e in events]
        rec.expect(lots, f"{len(lots)} lot(s) en quarantaine : {', '.join(lots)}", "aucun lot en quarantaine")
        failed = _failed_expectations(events)
        custom = sorted(failed & CUSTOM_EXPECTATIONS)
        rec.expect(custom, f"expectations custom en échec : {', '.join(custom)}",
                   f"aucune expectation custom en échec ({sorted(failed)})")  # fmt: skip
        rec.expect(_stored(ctx, lots) == 0, "aucun match de ces lots stocké en base")
        trainings = stack.flow_runs([TRAINING_FLOW], start)
        rec.expect(not stack.events("training_requested", start) and not trainings,
                   "aucun entraînement demandé ni lancé",
                   f"entraînement lancé : {[r.name for r in trainings]}")  # fmt: skip
        _alert(ctx, "quarantaine", flow_run_id=outcome.flow_run_id)
        _training_refused(ctx)
        rec.expect(stack.alias(Alias.PRODUCTION) == production, f"production inchangée (v{production})")
        _recover(ctx, lots)


class S3Retries(Scenario):
    """Pannes passagères de la source : les retries de Prefect, puis l'alerte quand ils ne suffisent plus."""

    id = "S3"
    title = "Source en 503 / slow"
    expected = "retries visibles dans Prefect, succès final ; alerte « dégradé » après épuisement des retries"

    def run(self, ctx: Context) -> None:
        """503 deux fois, lenteur une fois, puis 503 sans fin."""
        stack, rec = ctx.stack, ctx.rec
        retries = stack.config.ingestion.retries
        outcome = _ingest_with(ctx, "503", 7, failures=2)
        rec.expect(not outcome.failed and outcome.state_name == "Completed",
                   f"503 deux fois : ingestion réussie au 3e essai ({outcome.state_name})",
                   f"503 deux fois : {outcome.state_name}")  # fmt: skip
        retried = self._retried(ctx, outcome)
        rec.expect(retried and retried[0].run_count == 3,
                   f"retries visibles dans Prefect : « {retried[0].name if retried else '?'} » joué "
                   f"{retried[0].run_count if retried else 0} fois", "aucune task rejouée dans Prefect")  # fmt: skip

        timeout = stack.config.ingestion.http_timeout_s
        outcome = _ingest_with(ctx, "slow", 7, failures=1, delay_s=timeout + 2)
        retried = self._retried(ctx, outcome)
        rec.expect(not outcome.failed and retried and retried[0].run_count == 2,
                   f"source lente ({timeout + 2:.0f} s, délai {timeout:.0f} s) : 1er appel abandonné, réussite au "
                   f"2e essai (« {retried[0].name if retried else '?'} »)",
                   f"source lente : {outcome.state_name}, {[(t.name, t.run_count) for t in retried]}")  # fmt: skip

        start = now()
        outcome = _ingest_with(ctx, "503", 7)
        retried = self._retried(ctx, outcome)
        tries = max((t.run_count for t in retried), default=0)
        rec.expect(outcome.state_name == DEGRADED and tries == retries + 1,
                   f"503 sans fin : {tries} essais ({retries} retries) puis état « {DEGRADED} »",
                   f"503 sans fin : {outcome.state_name}, {tries} essai(s)")  # fmt: skip
        _alert(ctx, "degrade", flow_run_id=outcome.flow_run_id, since=start)

    @staticmethod
    def _retried(ctx: Context, outcome: FlowOutcome) -> list[Any]:
        if not outcome.flow_run_id:
            return []
        tasks = ctx.stack.task_runs(outcome.flow_run_id)
        return sorted((t for t in tasks if t.run_count > 1), key=lambda t: -t.run_count)


class S4SourceDown(Scenario):
    """La source ne répond plus du tout : on sert le dernier état validé et on prévient."""

    id = "S4"
    title = "Source arrêtée (docker compose stop mock_source)"
    expected = "run marqué « Dégradé », alerte ; watermark inchangé, l'API sert toujours le dernier état validé"

    def run(self, ctx: Context) -> None:
        """`mock_source` arrêté le temps d'une ingestion, puis relancé et son horloge remise à l'heure."""
        stack, rec = ctx.stack, ctx.rec
        clock, watermark = stack.clock(), stack.watermark()
        start = now()
        ctx.services.stop("mock_source")
        try:
            stack.wait_down("mock_source")
            rec.step("mock_source arrêté (docker compose stop mock_source)")
            outcome = ctx.actions.ingest()
            if outcome.flow_run_id:
                rec.link("Prefect : ingestion, source arrêtée", stack.prefect_run_url(outcome.flow_run_id))
            retries = stack.config.ingestion.retries
            rec.expect(outcome.state_name == DEGRADED,
                       f"ingestion terminée à l'état « {DEGRADED} » après {retries + 1} essais (code de sortie 2)",
                       f"ingestion : {outcome.state_name}")  # fmt: skip
            rec.expect(stack.watermark() == watermark, f"watermark inchangé ({watermark}) : rien n'est perdu")
            degraded = [e for e in stack.events("ingestion_degraded", start) if e["flow_run_id"] == outcome.flow_run_id]
            rec.expect(degraded, "incident inscrit dans pipeline_events (annotation Grafana)")
            _alert(ctx, "degrade", flow_run_id=outcome.flow_run_id)
            response = stack.predict(MATCH)
            status, ready = stack.ready()
            rec.expect(response.status_code == 200 and status == 200,
                       f"l'API sert toujours : données validées jusqu'au {ready['checks'].get('features_last_day')}, "
                       f"Djokovic contre Nadal {_probability(response)}",
                       f"API : prédiction HTTP {response.status_code}, /ready {status}")  # fmt: skip
        finally:
            ctx.services.start("mock_source")
            stack.wait_healthy("mock_source")
            stack.set_clock(clock)  # l'horloge vit en mémoire : un redémarrage la remet au début du rejeu
        rec.expect(stack.clock() == clock, f"mock_source relancé, horloge remise au {clock}")


class S5Schema(Scenario):
    """Le contrat de la source change : le contrôle de schéma bloque le lot."""

    id = "S5"
    title = "Schéma modifié (schema)"
    expected = "contrôle de schéma bloquant : lot en quarantaine, rien de stocké, alerte"

    def run(self, ctx: Context) -> None:
        """Une semaine ingérée pendant que la source renomme, supprime et retype des champs."""
        stack, rec = ctx.stack, ctx.rec
        start = now()
        outcome = _ingest_with(ctx, "schema", 7)
        rec.expect(outcome.failed, f"ingestion en échec (état {outcome.state_name})",
                   f"ingestion : {outcome.state_name}, pas en échec")  # fmt: skip
        events = _quarantined(ctx, start, outcome.flow_run_id)
        lots = [str(e["lot_id"]) for e in events]
        rec.expect(lots, f"{len(lots)} lot(s) en quarantaine : {', '.join(lots)}", "aucun lot en quarantaine")
        missing = sorted({str(c.get("target")) for e in events for c in e["payload"].get("failed_checks", [])
                          if c.get("expectation") == SCHEMA_EXPECTATION})  # fmt: skip
        rec.expect(missing, f"contrôle de schéma en échec : colonnes absentes {', '.join(missing)}",
                   f"pas d'échec du contrôle de schéma ({sorted(_failed_expectations(events))})")  # fmt: skip
        rec.expect(_stored(ctx, lots) == 0, "aucun match de ces lots stocké en base")
        _alert(ctx, "quarantaine", flow_run_id=outcome.flow_run_id)
        _recover(ctx, lots)
        settle(stack, start)


class S6WeakModel(Scenario):
    """Un modèle moins bon est enregistré, puis refusé par la règle de mise en service."""

    id = "S6"
    title = "Modèle moins bon"
    expected = "gate_status=failed, production inchangé"

    def run(self, ctx: Context) -> None:
        """`make train-weak` : régularisation extrême, probabilités écrasées vers 50 %."""
        stack, rec = ctx.stack, ctx.rec
        production, served = stack.alias(Alias.PRODUCTION), stack.served_version()
        rec.step(f"production avant : v{production}")
        code = ctx.actions.train_weak()
        rec.expect(code == 0, "campagne volontairement mauvaise terminée (make train-weak)", f"campagne : code {code}")
        weak = stack.latest_version() or "?"
        tags = stack.version_tags(weak)
        rec.link(f"MLflow : v{weak}", stack.mlflow_version_url(weak))
        rec.expect(weak != production, f"version enregistrée : v{weak}")
        rec.expect(tags.get("gate_status") == "failed" and tags.get("lifecycle") == "rejected",
                   f"gate_status=failed, lifecycle=rejected : {tags.get('gate_reason', '')[:160]}",
                   f"v{weak} : gate_status={tags.get('gate_status')}, lifecycle={tags.get('lifecycle')}")  # fmt: skip
        rec.expect(stack.alias(Alias.PRODUCTION) == production, f"production inchangée (v{production})")
        rec.expect(stack.alias(Alias.STAGING) == weak, f"v{weak} gardée en observation (alias staging)")
        rec.expect(stack.served_version() == served, f"l'API sert toujours v{served}")


class S7Rollback(Scenario):
    """Retour à la version précédente, servi sans redémarrer l'API."""

    id = "S7"
    title = "Rollback"
    expected = "production = ancienne version, l'API la sert sans redémarrage"

    def run(self, ctx: Context) -> None:
        """`make rollback` : production ← previous."""
        stack, rec = ctx.stack, ctx.rec
        production, previous = stack.alias(Alias.PRODUCTION), stack.alias(Alias.PREVIOUS)
        rec.expect(previous, f"avant : production v{production}, previous v{previous}",
                   "aucune version previous : S1 doit avoir promu une version")  # fmt: skip
        reloads = int(stack.model_info()["registry"]["reloads"])
        start = now()
        code = ctx.actions.rollback("S7 : test d'acceptation du rollback (make scenarios)")
        rec.expect(code == 0, "make rollback REASON=… accepté", f"make rollback : code {code}")
        rec.expect(stack.alias(Alias.PRODUCTION) == previous, f"alias production : v{production} → v{previous}")
        tags = stack.version_tags(str(production))
        rec.expect(tags.get("lifecycle") == "rolled_back", f"v{production} marquée rolled_back, avec la raison")
        wait_for(lambda: stack.served_version() == previous, API_REFRESH_TIMEOUT_S, f"API sur v{previous}")
        info = stack.model_info()
        response = stack.predict(MATCH)
        rec.expect(response.headers.get("x-model-version") == previous and int(info["registry"]["reloads"]) > reloads,
                   f"l'API sert v{previous} sans redémarrage (relecture de l'alias n° {info['registry']['reloads']})",
                   f"l'API sert v{response.headers.get('x-model-version')}")  # fmt: skip
        _alert(ctx, "rollback", since=start)


class S8MlflowDown(Scenario):
    """Le registre tombe : l'API continue de servir ce qu'elle a en mémoire, et trace les requêtes invalides."""

    id = "S8"
    title = "MLflow arrêté + entrée invalide"
    expected = "l'API sert le modèle en cache, /ready dégradé ; 422 tracé dans errors"

    def run(self, ctx: Context) -> None:
        """`mlflow` arrêté le temps des vérifications, puis relancé."""
        stack, rec = ctx.stack, ctx.rec
        served = stack.served_version()
        ctx.services.stop("mlflow")
        try:
            stack.wait_down("mlflow")
            rec.step("MLflow arrêté (docker compose stop mlflow)")
            status, body = wait_for(lambda: self._degraded(stack), API_REFRESH_TIMEOUT_S, "/ready dégradé")
            reason = next((r for r in body.get("reasons", []) if "MLflow" in r), "")
            rec.expect(status == 200, f"/ready : {body['status']} — {reason[:150]}")
            response = stack.predict(MATCH)
            rec.expect(response.status_code == 200 and response.headers.get("x-model-version") == served,
                       f"prédiction servie depuis le cache : v{served} (Djokovic-Nadal {_probability(response)})",
                       f"prédiction : HTTP {response.status_code}")  # fmt: skip
            invalid = stack.predict(INVALID_MATCH)
            request_id = invalid.headers.get("x-request-id", "")
            rec.expect(invalid.status_code == 422, f"entrée invalide (surface « Moon ») : HTTP 422 ({request_id})",
                       f"entrée invalide : HTTP {invalid.status_code}")  # fmt: skip
            row = wait_for(lambda: self._error(stack, request_id), 15, "ligne errors", interval_s=1)
            rec.expect(
                row["status_code"] == 422, f"422 tracé dans errors : {row['error_type']} — {row['message'][:100]}"
            )
        finally:
            ctx.services.start("mlflow")
            stack.wait_healthy("mlflow", timeout_s=240)
        wait_for(lambda: stack.ready()[1].get("status") == "ready", API_REFRESH_TIMEOUT_S, "/ready de nouveau prêt")
        rec.step("MLflow relancé : /ready de nouveau « ready »")

    @staticmethod
    def _degraded(stack: Stack) -> tuple[int, dict[str, Any]] | None:
        status, body = stack.ready()
        return (status, body) if body.get("status") == "degraded" else None

    @staticmethod
    def _error(stack: Stack, request_id: str) -> dict[str, Any] | None:
        rows = stack.rows("SELECT status_code, error_type, message FROM errors WHERE request_id = :r", r=request_id)
        return rows[0] if rows else None


class S9Interface(Scenario):
    """L'interface de test, utilisée comme un visiteur."""

    id = "S9"
    title = "Interface de test"
    expected = "l'UI répond, une prédiction faite depuis l'UI apparaît dans predictions avec X-Client: ui"

    def run(self, ctx: Context) -> None:
        """La page Match prédit deux joueurs ; la trace est relue en base (`make ui-smoke`)."""
        stack, rec = ctx.stack, ctx.rec
        rec.link("Interface de test", stack.url("ui"))
        try:
            steps = ctx.actions.ui_smoke(stack.url("ui"), stack.settings)
        except AssertionError as exc:
            rec.expect(False, "", f"interface : {exc}")
            return
        for step in steps:
            rec.step(step)


SCENARIOS: tuple[Callable[[], Scenario], ...] = (S1Nominal, S2Corrupt, S3Retries, S4SourceDown, S5Schema,
                                                  S6WeakModel, S7Rollback, S8MlflowDown, S9Interface)  # fmt: skip


def scenario_ids() -> list[str]:
    """`S1` … `S9`."""
    return [factory().id for factory in SCENARIOS]


def replay_day(stack: Stack) -> date:
    """Début du rejeu (l'horloge de l'état de référence)."""
    return stack.config.simulation.replay_start
