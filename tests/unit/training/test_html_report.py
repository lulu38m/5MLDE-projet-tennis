import math

import pandas as pd
from matplotlib.figure import Figure

from tennis_mlops.training import html_report as h


def test_markdown_of_the_model_card() -> None:
    text = "# Titre\n\n## Données\n\n- un **gras** et du `code`\n- <script>\n\n| a | b |\n|---|---:|\n| 1 | 2 |\n\nFin."
    rendered = h.markdown(text)
    assert "<h3>Données</h3>" in rendered
    assert "Titre" not in rendered  # le titre de la fiche est celui de la section
    assert "<li>un <strong>gras</strong> et du <code>code</code></li>" in rendered
    assert "&lt;script&gt;" in rendered and "<script>" not in rendered
    assert "<thead><tr><th>a</th><th>b</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody>" in rendered
    assert rendered.endswith("<p>Fin.</p>")


def test_tables_format_numbers_missing_values_and_highlight() -> None:
    frame = pd.DataFrame(
        {
            "nom": ["<b>x</b>", "y"],
            "score": [0.123456, math.nan],
            "jour": pd.to_datetime(["2016-01-02", "2016-01-03"]),
            "retenu": [True, False],
            "classement": [370.0, math.nan],
        }
    )
    rendered = h.table(frame, digits=3, highlight="retenu", scroll=True)
    assert '<div class="scroll">' in rendered
    assert '<th class="num">score</th>' in rendered
    assert "0.123" in rendered and "—" in rendered and "2016-01-02" in rendered
    assert ">370<" in rendered  # un classement reste un entier, même stocké en flottant
    assert "&lt;b&gt;x&lt;/b&gt;" in rendered
    assert rendered.count('class="selected"') == 1
    assert "retenu" not in rendered


def test_cards_images_and_page() -> None:
    figure = Figure()
    figure.subplots().plot([0, 1], [1, 0])
    assert h.figure_src(figure).startswith("data:image/png;base64,")
    body = [h.cards([h.Card("log loss", "0.60", "-0.01 contre l'Elo", True), h.Card("ECE", "0.02")]),
            h.section("Graphiques", h.images({"courbe": figure}))]  # fmt: skip
    document = h.page("Rapport <test>", "sous-titre", body)
    assert document.startswith("<!doctype html>")
    assert "<title>Rapport &lt;test&gt;</title>" in document
    assert 'class="delta good"' in document
    assert "<h2>Graphiques</h2>" in document and 'alt="courbe"' in document
