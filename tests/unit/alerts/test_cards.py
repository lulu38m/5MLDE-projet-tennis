import json
from datetime import UTC, datetime

import pytest

from tennis_mlops.alerts.cards import Alert, Tone, adaptive_card, card_texts, display_time, teams_message
from tennis_mlops.alerts.examples import EXAMPLES


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_every_example_is_a_valid_teams_message(name: str) -> None:
    message = teams_message(EXAMPLES[name]())
    assert message["type"] == "message"
    (attachment,) = message["attachments"]
    assert attachment["contentType"] == "application/vnd.microsoft.card.adaptive"
    card = attachment["content"]
    assert (card["type"], card["version"], card["msteams"]) == ("AdaptiveCard", "1.5", {"width": "Full"})
    json.dumps(message)  # sérialisable tel quel
    texts = card_texts(card)
    assert not any("`" in t for t in texts)  # Teams n'affiche pas le code en ligne
    header = card["body"][0]
    assert header["type"] == "Container" and header["bleed"] is True


def test_layout_title_emoji_only_and_sections() -> None:
    alert = Alert(
        tone=Tone.FAILURE,
        title="❌ Échec de l'entraînement",
        subtitle="entrainement-atp · run",
        status="État Failed",
        reason="partition `train` trop petite",
        highlights=(("Étape en erreur", "TemporalSplitter.split"), ("Lot", "2016-520"), ("Suite", "v1.0"), ("x", "y")),
        facts=(("Flow run", "abc"),),
        next_steps="relancer `make ingest`",
        links=(("Ouvrir dans Prefect", "http://127.0.0.1:4201/runs/flow-run/abc"),),
        host="poste",
        at=datetime(2026, 11, 12, 11, 6, tzinfo=UTC),
    )
    card = adaptive_card(alert)
    header, cause_title, cause, columns, facts_title, facts, todo_title, _todo, footer = card["body"]
    assert header["style"] == "attention"
    assert [i["text"] for i in header["items"]] == [
        "❌ Échec de l'entraînement",
        "entrainement-atp · run",
        "État Failed",
    ]
    assert (cause_title["text"], todo_title["text"], facts_title["text"]) == (
        "Cause",
        "Que faire",
        "Retrouver l'exécution",
    )
    assert cause["text"] == "partition «\N{NO-BREAK SPACE}train\N{NO-BREAK SPACE}» trop petite"
    assert len(columns["columns"]) == 3  # au plus trois chiffres clés
    sizes = [c["items"][1]["size"] for c in columns["columns"]]
    assert sizes == ["Medium", "Large", "Large"]  # une valeur longue ne se coupe pas au milieu d'un mot
    assert facts["facts"] == [{"title": "Flow run", "value": "abc"}]
    assert footer["text"] == "poste · 12/11/2026 à 12:06 · liens locaux : ils ne s'ouvrent que sur ce poste"
    assert card["actions"] == [{"type": "Action.OpenUrl", "title": "Ouvrir dans Prefect",
                                "url": "http://127.0.0.1:4201/runs/flow-run/abc"}]  # fmt: skip


def test_minimal_card_and_paris_time() -> None:
    card = adaptive_card(Alert(Tone.INFO, "🔵 Info", "sujet", "état", "raison"))
    assert [b["type"] for b in card["body"]] == ["Container", "TextBlock", "TextBlock", "TextBlock"]
    assert "actions" not in card
    assert card["body"][-1]["text"] == "liens locaux : ils ne s'ouvrent que sur ce poste"
    assert display_time(datetime(2026, 7, 1, 22, 30, tzinfo=UTC)) == "02/07/2026 à 00:30"  # heure d'été
