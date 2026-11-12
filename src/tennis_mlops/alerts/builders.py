"""Des faits du pipeline aux cartes : échec d'un flow, lot en quarantaine, ingestion dégradée, rollback, promotion.

Les fonctions sont pures : elles reçoivent ce que le service d'alertes a lu (run Prefect, événements de
`pipeline_events`, bilan des lots, décision du registre) et rendent une `Alert`. Les tests les appellent directement ;
`examples.py` montre le rendu attendu de chacune.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from tennis_mlops.alerts.cards import Alert, Tone, display_time

DEGRADED_STATE = "Dégradé"
TRAINING_FLOW = "entrainement-atp"
FLOW_LABELS = {
    "ingestion-atp": "l'ingestion ATP",
    TRAINING_FLOW: "l'entraînement",
    "simulation-atp": "la simulation",
}
NEXT_STEPS = {
    "ingestion-atp": "Les données déjà en base restent utilisables. Lire la cause dans Prefect, corriger, puis "
    "relancer `make ingest` : les lots déjà stockés ne sont pas refaits.",
    TRAINING_FLOW: "Rien ne change côté service : l'API garde la version en production. Lire la cause dans "
    "Prefect, corriger, puis relancer `make train`.",
    "simulation-atp": "`make demo-stop` suspend la simulation le temps de corriger ; `make demo` la relance.",
}
CHECK_LABELS = {
    "expect_tennis_score_to_be_valid": "Score lisible",
    "expect_winner_and_loser_to_differ": "Vainqueur ≠ perdant",
    "expect_no_match_after_simulated_now": "Aucun match après l'horloge simulée",
    "expect_batch_volume_to_match_draw": "Volume = taille du tableau",
    "expect_table_row_count_to_be_between": "Lot non vide",
    "expect_column_to_exist": "Colonne présente",
    "expect_column_values_to_not_be_null": "Valeur obligatoire",
    "expect_column_values_to_be_unique": "Identifiant unique",
    "expect_column_values_to_be_in_type_list": "Type attendu",
    "expect_column_values_to_be_in_set": "Valeur autorisée",
    "expect_column_values_to_match_regex": "Format attendu",
    "expect_column_values_to_be_between": "Valeur dans les bornes",
}
LEVELS = {
    "G": "Grand Chelem",
    "M": "Masters 1000",
    "A": "ATP 250/500",
    "F": "Masters de fin d'année",
    "D": "Coupe Davis",
    "O": "Jeux olympiques",
}
MAX_SAMPLES = 2
MAX_CHECK_LINES = 5
MISSING_COLUMN = "expect_column_to_exist"


class AlertKind(StrEnum):
    """Les familles de cartes (ce sont les noms de `orchestration.teams_events`)."""

    FAILURE = "echec"
    QUARANTINE = "quarantaine"
    DEGRADED = "degrade"
    ROLLBACK = "rollback"
    PROMOTION = "promotion"


@dataclass(frozen=True)
class CardContext:
    """Ce qui est commun à toutes les cartes : qui envoie, adresses des interfaces, heure."""

    host: str
    prefect_ui: str
    mlflow_ui: str
    at: datetime
    owner: str = ""

    @property
    def project(self) -> tuple[str, str]:
        """La ligne « Projet » : dans un canal partagé avec la classe, on sait d'où vient la carte."""
        who = f" · {self.owner}" if self.owner else ""
        return ("Projet", f"tennis-mlops{who} · poste {self.host}")

    def prefect_run(self, flow_run_id: str) -> str:
        """Lien vers un flow run dans l'UI Prefect."""
        return f"{self.prefect_ui.rstrip('/')}/runs/flow-run/{flow_run_id}"

    def mlflow_run(self, run_id: str, experiment_id: str) -> str:
        """Lien vers un run dans l'UI MLflow."""
        return f"{self.mlflow_ui.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"

    def mlflow_model(self, name: str) -> str:
        """Lien vers un modèle du registre."""
        return f"{self.mlflow_ui.rstrip('/')}/#/models/{name}"


@dataclass(frozen=True)
class FlowRunInfo:
    """Ce que le service d'alertes lit d'un flow run dans l'API Prefect."""

    id: str
    name: str
    flow_name: str
    state_type: str  # FAILED, CRASHED, COMPLETED…
    state_name: str  # Failed, Crashed, Dégradé…
    message: str = ""
    failed_task: str | None = None
    deployment: str | None = None
    started_at: datetime | None = None
    tags: Sequence[str] = ()
    automation: bool = False  # lancé par une automation (événement `tennis.training.requested`)


@dataclass(frozen=True)
class PrefectEvent:
    """L'événement Prefect qui a déclenché l'automation (ce que poste le webhook)."""

    id: str
    name: str  # prefect.flow-run.Failed…

    def line(self) -> str:
        """Ligne « Événement » de la carte : si une carte arrive deux fois, on voit que c'est la même."""
        return f"{_short(self.id)} ({self.name}), pour reconnaître un doublon"


@dataclass(frozen=True)
class CampaignLink:
    """Run de campagne MLflow lié à un flow run d'entraînement."""

    run_id: str
    experiment_id: str
    status: str


@dataclass(frozen=True)
class Situation:
    """Ce que la base sait au moment de l'alerte : version servie, données disponibles."""

    production: str | None = None
    data_end: str | None = None  # fin du dernier lot validé (ISO)
    watermark: str | None = None
    lots_valid: int | None = None


NO_SITUATION = Situation()


class RegistryNotice(BaseModel):
    """Une décision du registre envoyée au service d'alertes (`RegistryEvent.as_dict()` + quelques chiffres)."""

    kind: str  # promotion, rejet, promotion manuelle, rollback
    model: str
    version: str
    production_before: str | None = None
    production_after: str | None = None
    decided_by: str = "pipeline"
    reason: str = ""
    at: str = ""
    campaign_run_id: str | None = None
    gate: dict[str, Any] | None = None
    metrics: dict[str, Any] = {}  # chiffres de la campagne (flows/train.py) ou `previous_after` (rollback)

    @property
    def ref(self) -> str:
        """Identifiant stable : une même décision n'est notifiée qu'une fois."""
        return f"registre:{self.model}:{self.kind}:{self.version}:{self.at}"[:64]


def classify(info: FlowRunInfo, quarantined: Sequence[Mapping[str, Any]]) -> AlertKind:
    """Quelle carte pour ce flow run : dégradé, quarantaine ou échec."""
    if info.state_name == DEGRADED_STATE:
        return AlertKind.DEGRADED
    if quarantined:
        return AlertKind.QUARANTINE
    return AlertKind.FAILURE


# --------------------------------------------------------------------------- flows


def failure_alert(
    info: FlowRunInfo,
    event: PrefectEvent,
    ctx: CardContext,
    situation: Situation = NO_SITUATION,
    campaign: CampaignLink | None = None,
) -> Alert:
    """Un flow fini en `Failed` ou `Crashed` après tous ses retries."""
    crashed = info.state_type.upper() == "CRASHED"
    label = FLOW_LABELS.get(info.flow_name, f"« {info.flow_name} »")
    title = f"💥 Arrêt brutal de {label}" if crashed else f"❌ Échec de {label}"
    production = f"v{situation.production} inchangée" if situation.production else "aucune"
    if info.flow_name == TRAINING_FLOW:
        middle = ("Données jusqu'au", _fr_date(situation.data_end))
    else:
        middle = ("Lancé par", launcher(info))
    facts: list[tuple[str, str]] = [("Flow run", info.id)]
    links = [("Ouvrir le run dans Prefect", ctx.prefect_run(info.id))]
    if campaign is not None:
        facts.append(("Campagne MLflow", f"{_short(campaign.run_id)} ({campaign.status})"))
        links.append(("Ouvrir la campagne dans MLflow", ctx.mlflow_run(campaign.run_id, campaign.experiment_id)))
    facts += [ctx.project, ("Événement", event.line())]
    started = f" · démarré le {display_time(info.started_at)}" if info.started_at else ""
    return Alert(
        tone=Tone.FAILURE,
        title=title,
        subtitle=f"{info.flow_name} · {info.name}{started}",
        status=f"État {info.state_name} · un incident résolu par les retries ne déclenche rien",
        reason=info.message.strip() or "non renseignée (voir les logs du run)",
        highlights=(("Étape en erreur", info.failed_task or "hors task"), middle, ("Production", production)),
        facts=tuple(facts),
        next_steps=NEXT_STEPS.get(info.flow_name, "Lire la cause dans Prefect, corriger, puis relancer."),
        links=tuple(links),
        host=ctx.host,
        at=ctx.at,
    )


def quarantine_alert(
    info: FlowRunInfo, lots: Sequence[Mapping[str, Any]], event: PrefectEvent, ctx: CardContext, suite: str
) -> Alert:
    """Great Expectations a refusé un ou plusieurs lots (payloads des événements `lot_quarantined`)."""
    first = lots[0]
    lot_id = str(first.get("lot_id"))
    about = " · ".join(str(p) for p in (first.get("tourney_name"), LEVELS.get(str(first.get("tourney_level")))) if p)
    subtitle = f"{info.flow_name} · " + (f"{about} (lot {lot_id})" if about else f"lot {lot_id}")
    received, expected = first.get("rows_received"), first.get("rows_expected")
    reason = check_lines(first.get("failed_checks") or [])
    others = [str(lot.get("lot_id")) for lot in lots[1:]]
    if others:
        reason += f"\n\nAutres lots refusés pendant ce passage : {', '.join(others)}."
    return Alert(
        tone=Tone.FAILURE,
        title="🚫 Lot refusé par Great Expectations",
        subtitle=subtitle,
        status=f"État {info.state_name} · lot archivé en quarantaine, aucun match écrit en base",
        reason_title="Contrôles en échec (suite bloquante)",
        reason=reason,
        highlights=(
            ("Lot", lot_id),
            ("Matchs reçus", f"{received} sur {expected}" if expected is not None else str(received)),
            ("Suite GX", f"v{suite}"),
        ),
        facts=(
            ("Flow run", info.id),
            ("Quarantaine", str(first.get("archive") or f"state/quarantine/{lot_id}/")),
            ("Rapport détaillé", "make gx-docs"),
            ctx.project,
            ("Événement", event.line()),
        ),
        next_steps="Le watermark n'a pas avancé : l'entraînement ne voit pas ce lot. Corriger la source, puis "
        "`make ingest` rejoue le lot ; `make lots` montre son statut.",
        links=(("Ouvrir le run dans Prefect", ctx.prefect_run(info.id)),),
        host=ctx.host,
        at=ctx.at,
    )


def degraded_alert(info: FlowRunInfo, situation: Situation, event: PrefectEvent, ctx: CardContext) -> Alert:
    """La source ne répond plus après tous les essais : rien n'est faux, la chaîne attend."""
    return Alert(
        tone=Tone.WARNING,
        title="⚠️ Ingestion dégradée : source injoignable",
        subtitle=f"{info.flow_name} · {info.name}",
        status="État Dégradé · tous les essais épuisés, la chaîne attend la source",
        reason=info.message.strip() or "Source indisponible.",
        highlights=(
            ("Données en base jusqu'au", _fr_date(situation.data_end)),
            ("Ingéré jusqu'au", _fr_date(situation.watermark)),
            ("Lots validés", "—" if situation.lots_valid is None else str(situation.lots_valid)),
        ),
        facts=(("Flow run", info.id), ctx.project, ("Événement", event.line())),
        next_steps="Les données déjà en base restent utilisables (entraînement, API). Vérifier mock_source "
        "(`make logs`) ; la prochaine exécution planifiée reprendra au watermark.",
        links=(("Ouvrir le run dans Prefect", ctx.prefect_run(info.id)),),
        host=ctx.host,
        at=ctx.at,
    )


# --------------------------------------------------------------------------- registre


def rollback_alert(notice: RegistryNotice, ctx: CardContext) -> Alert:
    """Une personne a remis la version précédente en production."""
    bad, good = notice.production_before, notice.production_after
    following = notice.metrics.get("previous_after")
    return Alert(
        tone=Tone.WARNING,
        title=f"↩️ Rollback : production v{bad} → v{good}",
        subtitle=f"{notice.model} · décidé à la main (make rollback)",
        status=f"L'API suit l'alias production : elle sert v{good} sans redéploiement",
        reason_title="Raison",
        reason=notice.reason or "non précisée",
        highlights=(
            ("Retirée (rolled_back)", f"v{bad}"),
            ("Servie", f"v{good}"),
            ("Rollback suivant", f"vers v{following}" if following else "impossible"),
        ),
        facts=(("Journal", f"rollback v{bad} → v{good} ({notice.model}-decisions)"), ctx.project),
        next_steps=f"Analyser v{bad} (`make trace VERSION={bad}`) ; elle reste dans le registre, `make promote` "
        "la remettrait en service.",
        links=(("Ouvrir le registre dans MLflow", ctx.mlflow_model(notice.model)),),
        host=ctx.host,
        at=ctx.at,
    )


def promotion_alert(notice: RegistryNotice, ctx: CardContext) -> Alert:
    """Une nouvelle version passe en production (règle de mise en service, ou décision manuelle)."""
    manual = notice.kind == "promotion manuelle"
    m = notice.metrics
    before = f"remplace v{notice.production_before}" if notice.production_before else "première version servie"
    if manual:
        subtitle, status = f"{before} · décidé à la main (make promote)", "Décision manuelle : la règle est contournée"
        reason_title, reason = "Raison", notice.reason or "non précisée"
    else:
        rules = (notice.gate or {}).get("rules", [])
        passed = sum(1 for r in rules if r.get("passed"))
        when = f" du {_fr_date(notice.at)[:5]}" if notice.at else ""
        count = f" ({m['candidates']} candidats)" if m.get("candidates") else ""
        subtitle = f"{before} · campagne entrainement-atp{when}{count}"
        status = f"Règle de mise en service : {passed} règles sur {len(rules)}"
        reason_title, reason = "Sur le même test (6 derniers mois)", _comparison(notice)
    facts: list[tuple[str, str]] = []
    if notice.campaign_run_id:
        facts.append(("Campagne MLflow", notice.campaign_run_id))
    if m.get("data_end"):
        facts.append(("Données jusqu'au", _fr_date(m["data_end"])))
    facts.append(ctx.project)
    links: list[tuple[str, str]] = []
    if notice.campaign_run_id and m.get("experiment_id"):
        campaign = ctx.mlflow_run(notice.campaign_run_id, str(m["experiment_id"]))
        links.append(("Rapport de campagne", f"{campaign}/artifacts/rapport_campagne.html"))
    links.append(("Ouvrir le registre dans MLflow", ctx.mlflow_model(notice.model)))
    return Alert(
        tone=Tone.SUCCESS,
        title=f"✅ Nouveau modèle en production : {notice.model} v{notice.version}",
        subtitle=subtitle,
        status=status,
        reason_title=reason_title,
        reason=reason,
        highlights=(
            ("Log loss", _decimal(m.get("log_loss"))),
            ("Accuracy", _percent(m.get("accuracy"))),
            ("Modèle", str(m.get("candidate") or "—")),
        ),
        facts=tuple(facts),
        links=tuple(links),
        host=ctx.host,
        at=ctx.at,
    )


# --------------------------------------------------------------------------- mise en forme


def check_lines(checks: Sequence[Mapping[str, Any]]) -> str:
    """Contrôles en échec (`CheckResult` en dict) → liste Markdown courte, la cause d'abord.

    Une colonne absente fait échouer tous les contrôles qui la lisent : on dit quelles colonnes manquent, puis on
    compte ces échecs par conséquence au lieu de les lister un par un. Au-delà de `MAX_CHECK_LINES`, on renvoie au
    rapport Great Expectations.
    """
    missing = [str(c.get("target")) for c in checks if c.get("expectation") == MISSING_COLUMN]
    lines = [f"- **Colonnes absentes** : {', '.join(missing)}"] if missing else []
    consequences = 0
    for check in checks:
        if check.get("expectation") == MISSING_COLUMN:
            continue
        targets = set(str(check.get("target") or "").split("/"))
        if missing and targets & set(missing) and not check.get("unexpected_count"):
            consequences += 1
            continue
        lines.append(_check_line(check))
    shown = lines[:MAX_CHECK_LINES]
    if hidden := len(lines) - len(shown):
        shown.append(f"- … et {hidden} autre(s) contrôle(s) : détail dans `make gx-docs`")
    if consequences:
        shown.append(f"- {consequences} contrôle(s) sur ces colonnes, en échec par conséquence")
    return "\n".join(shown) or "- contrôle bloquant en échec (détail dans le rapport Great Expectations)"


def _check_line(check: Mapping[str, Any]) -> str:
    name = str(check.get("expectation", ""))
    label = CHECK_LABELS.get(name, name.removeprefix("expect_").replace("_", " "))
    target = str(check.get("target") or "")
    where = f" ({target})" if target and target != "table" else ""
    count = check.get("unexpected_count")
    samples = [str(s) for s in check.get("sample") or []]
    if check.get("error"):
        detail = str(check["error"])
    elif count:
        examples = ", ".join(f"`{s}`" for s in samples[:MAX_SAMPLES])
        detail = f"{count} match{'s' if int(count) > 1 else ''}" + (f", ex. {examples}" if examples else "")
    else:
        detail = ", ".join(samples[:MAX_SAMPLES])
    return f"- **{label}**{where}" + (f" : {detail}" if detail else "")


def launcher(info: FlowRunInfo) -> str:
    """Qui a lancé le run : planning, automation, déploiement ou commande locale."""
    if any(t.startswith("auto-scheduled") for t in info.tags):
        return "planning"
    if info.automation:
        return "automation"
    if info.deployment:
        return f"déploiement {info.deployment}"
    return "commande locale"


def _comparison(notice: RegistryNotice) -> str:
    m = notice.metrics
    against = []
    if m.get("production_log_loss") is not None and notice.production_before:
        against.append(f"{_decimal(m['production_log_loss'])} pour v{notice.production_before}")
    if m.get("elo_log_loss") is not None:
        against.append(f"{_decimal(m['elo_log_loss'])} pour l'Elo pur")
    text = f"Log loss **{_decimal(m.get('log_loss'))}**" + (" contre " + " et ".join(against) if against else "")
    rules = (notice.gate or {}).get("rules", [])
    segments = next((r for r in rules if r.get("rule") == "segments"), None)
    if segments is not None and segments.get("applicable", True) and segments.get("passed"):
        text += " ; aucune surface dégradée"
    return text + "."


def _fr_date(value: object) -> str:
    if not value:
        return "—"
    if isinstance(value, datetime | date):
        return value.strftime("%d/%m/%Y")
    text = str(value)[:10]
    try:
        return date.fromisoformat(text).strftime("%d/%m/%Y")
    except ValueError:
        return text


def _short(identifier: str) -> str:
    return identifier if len(identifier) <= 12 else f"{identifier[:4]}…{identifier[-3:]}"


def _decimal(value: object, digits: int = 4) -> str:
    return "—" if value is None else f"{float(str(value)):.{digits}f}".replace(".", ",")


def _percent(value: object) -> str:
    return "—" if value is None else f"{float(str(value)) * 100:.1f} %".replace(".", ",")
