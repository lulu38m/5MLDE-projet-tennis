"""Cartes Teams (Adaptive Card 1.5) des alertes du pipeline.

Une carte = un bandeau coloré (titre, de quoi il s'agit, état), la cause, trois chiffres clés, de quoi retrouver
l'exécution, la marche à suivre et des boutons vers Prefect ou MLflow. Seul le titre porte un emoji : il dit la
nature de l'alerte d'un coup d'œil, le reste reste sobre. Le message suit le format attendu par le
modèle Workflows « Envoyer des alertes webhook à un canal » : `{"type": "message", "attachments": [carte]}`.

Teams n'accepte qu'un Markdown réduit dans les cartes (gras, italique, listes, liens) : le code entre accents graves
s'afficherait tel quel, il est donc converti en « guillemets ».

Ce module ne fait que construire du JSON (aucun envoi, aucun secret) : il est testé tel quel et prévisualisé avec le
moteur de rendu officiel des Adaptive Cards avant tout envoi réel.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, cast
from zoneinfo import ZoneInfo

SCHEMA = "http://adaptivecards.io/schemas/adaptive-card.json"
CARD_VERSION = "1.5"
DISPLAY_ZONE = ZoneInfo("Europe/Paris")
LARGE_VALUE_CHARS = 14  # au-delà, un chiffre clé passe en taille moyenne (sinon il se coupe au milieu d'un mot)
INLINE_CODE = re.compile(r"`([^`]+)`")


class Tone(StrEnum):
    """Couleur du bandeau (styles de conteneur Adaptive Cards)."""

    FAILURE = "attention"  # rouge : quelque chose est cassé, il faut agir
    WARNING = "warning"  # orange : dégradé ou retour arrière, la chaîne tient encore
    SUCCESS = "good"  # vert : une bonne nouvelle (nouveau modèle en service)
    INFO = "accent"  # bleu : information


@dataclass(frozen=True)
class Alert:
    """Le contenu d'une carte, indépendant de sa mise en page."""

    tone: Tone
    title: str  # « ❌ Échec de l'entraînement »
    subtitle: str  # de quoi il s'agit : flow · run, ou modèle · version
    status: str  # une ligne discrète sous le titre : état, ce qui a déclenché l'alerte
    reason: str  # la cause ou la raison (Markdown simple : listes, **gras**)
    reason_title: str = "Cause"
    highlights: Sequence[tuple[str, str]] = ()  # jusqu'à 3 chiffres clés, affichés en colonnes
    facts: Sequence[tuple[str, str]] = ()  # identifiants pour retrouver l'exécution
    facts_title: str = "Retrouver l'exécution"
    next_steps: str = ""  # la marche à suivre (Markdown simple)
    links: Sequence[tuple[str, str]] = ()  # boutons (libellé, URL)
    host: str = ""
    at: datetime | None = None
    footer_note: str = "liens locaux : ils ne s'ouvrent que sur ce poste"


def adaptive_card(alert: Alert) -> dict[str, Any]:
    """La carte Adaptive (le `content` de la pièce jointe), textes adaptés au Markdown de Teams."""
    body: list[dict[str, Any]] = [_header(alert), *_section(alert.reason_title, alert.reason, separator=False)]
    if alert.highlights:
        body.append(_highlights(alert.highlights))
    if alert.facts:
        body.append(_title(alert.facts_title, separator=True))
        body.append(
            {"type": "FactSet", "spacing": "Small", "facts": [{"title": t, "value": v} for t, v in alert.facts]}
        )
    if alert.next_steps:
        body.extend(_section("Que faire", alert.next_steps, separator=True))
    body.append(_footer(alert))
    card: dict[str, Any] = {
        "$schema": SCHEMA,
        "type": "AdaptiveCard",
        "version": CARD_VERSION,
        "msteams": {"width": "Full"},
        "body": body,
    }
    if alert.links:
        card["actions"] = [{"type": "Action.OpenUrl", "title": t, "url": u} for t, u in alert.links]
    return cast(dict[str, Any], _teams_markdown(card))


def teams_message(alert: Alert) -> dict[str, Any]:
    """Le corps à poster au webhook Workflows."""
    return {
        "type": "message",
        "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": adaptive_card(alert)}],
    }


def display_time(at: datetime) -> str:
    """« 12/11/2026 à 12:06 » à l'heure de Paris (le canal est lu en France)."""
    return at.astimezone(DISPLAY_ZONE).strftime("%d/%m/%Y à %H:%M")


# --------------------------------------------------------------------------- blocs


def _header(alert: Alert) -> dict[str, Any]:
    return {
        "type": "Container",
        "style": alert.tone.value,
        "bleed": True,
        "items": [
            {"type": "TextBlock", "text": alert.title, "size": "Large", "weight": "Bolder", "wrap": True},
            {"type": "TextBlock", "text": alert.subtitle, "wrap": True, "spacing": "Small"},
            {"type": "TextBlock", "text": alert.status, "wrap": True, "spacing": "None", "isSubtle": True},
        ],
    }


def _title(text: str, separator: bool) -> dict[str, Any]:
    return {
        "type": "TextBlock",
        "text": text,
        "size": "Medium",
        "weight": "Bolder",
        "wrap": True,
        "separator": separator,
        "spacing": "Medium",
    }


def _section(title: str, text: str, separator: bool) -> list[dict[str, Any]]:
    return [_title(title, separator), {"type": "TextBlock", "text": text, "wrap": True, "spacing": "Small"}]


def _highlights(items: Sequence[tuple[str, str]]) -> dict[str, Any]:
    columns = [
        {
            "type": "Column",
            "width": "stretch",
            "items": [
                {"type": "TextBlock", "text": label, "isSubtle": True, "wrap": True},
                {
                    "type": "TextBlock",
                    "text": value,
                    "size": "Large" if len(value) <= LARGE_VALUE_CHARS else "Medium",
                    "weight": "Bolder",
                    "wrap": True,
                    "spacing": "Small",
                },
            ],
        }
        for label, value in items[:3]
    ]
    return {"type": "ColumnSet", "separator": True, "spacing": "Medium", "columns": columns}


def _footer(alert: Alert) -> dict[str, Any]:
    parts = [alert.host, display_time(alert.at) if alert.at else ""]
    text = " · ".join(p for p in [*parts, alert.footer_note] if p)
    return {
        "type": "TextBlock",
        "text": text,
        "wrap": True,
        "isSubtle": True,
        "size": "Small",
        "separator": True,
        "spacing": "Medium",
    }


def _teams_markdown(node: object) -> object:
    """`code` → « code » dans tous les textes (Teams n'affiche pas le code en ligne)."""
    if isinstance(node, dict):
        return {k: (INLINE_CODE.sub("«\u00a0\\1\u00a0»", v) if k in {"text", "title", "value"} and isinstance(v, str)
                    else _teams_markdown(v)) for k, v in node.items()}  # fmt: skip
    if isinstance(node, list):
        return [_teams_markdown(item) for item in node]
    return node


def card_texts(card: dict[str, Any]) -> list[str]:
    """Tous les textes d'une carte, dans l'ordre (tests, aperçu)."""
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key in ("text", "title", "value"):
                if isinstance(node.get(key), str):
                    found.append(node[key])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(card.get("body", []))
    walk(card.get("actions", []))
    return found
