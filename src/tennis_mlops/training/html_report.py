"""Rapports HTML autonomes, journalisés dans les artefacts MLflow (l'onglet « Artifacts » les affiche directement).

Un fichier = tout ce qu'il faut pour juger un candidat (ou une campagne) d'un coup d'œil : graphiques intégrés en
base64, tableaux, fiche modèle. Aucun JavaScript, aucune ressource externe : le fichier s'ouvre aussi hors MLflow.
"""

from __future__ import annotations

import base64
import html
import io
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

STYLE = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #1f2328;
       background: #ffffff; margin: 0; padding: 24px 32px 48px; line-height: 1.45; font-size: 14px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #d8dee4; }
h3 { font-size: 15px; margin: 20px 0 6px; }
p.lead { color: #59636e; margin: 0 0 16px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 16px 0; }
.card { border: 1px solid #d8dee4; border-radius: 8px; padding: 10px 14px; }
.card .label { color: #59636e; font-size: 12px; }
.card .value { font-size: 22px; font-weight: 600; }
.card .delta { font-size: 12px; }
.good { color: #1a7f37; } .bad { color: #cf222e; } .muted { color: #59636e; }
table { border-collapse: collapse; margin: 8px 0 16px; font-size: 13px; }
th, td { border-bottom: 1px solid #e6e9ed; padding: 4px 10px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.selected td { background: #fff4e5; font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
.grid img, img.wide { max-width: 100%; border: 1px solid #e6e9ed; border-radius: 6px; }
code { background: #f6f8fa; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
.scroll { max-height: 520px; overflow: auto; }
footer { margin-top: 40px; color: #59636e; font-size: 12px; }
"""


@dataclass(frozen=True)
class Card:
    """Un chiffre clé, avec son écart à une référence (négatif = mieux pour une perte)."""

    label: str
    value: str
    delta: str = ""
    good: bool | None = None


def figure_src(figure: Figure) -> str:
    """Le graphique en PNG, intégré au fichier (data URI)."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def page(title: str, lead: str, body: Iterable[str]) -> str:
    """Document HTML complet."""
    return (
        f'<!doctype html>\n<html lang="fr"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><style>{STYLE}</style></head>'
        f'<body><h1>{html.escape(title)}</h1><p class="lead">{lead}</p>{"".join(body)}'
        "<footer>Généré par tennis_mlops (projet 5MLDE). Les fichiers CSV et PNG du même run contiennent les mêmes "
        "données en version brute.</footer></body></html>\n"
    )


def section(title: str, *content: str) -> str:
    """Un titre de section et son contenu."""
    return f"<h2>{html.escape(title)}</h2>{''.join(content)}"


def paragraph(text: str) -> str:
    """Un paragraphe (texte déjà échappé ou construit avec `esc`)."""
    return f"<p>{text}</p>"


def esc(value: object) -> str:
    """Texte échappé pour HTML."""
    return html.escape(str(value))


def cards(items: Sequence[Card]) -> str:
    """Rangée de chiffres clés."""
    blocks = []
    for item in items:
        tone = "" if item.good is None else ("good" if item.good else "bad")
        delta = f'<div class="delta {tone}">{esc(item.delta)}</div>' if item.delta else ""
        blocks.append(f'<div class="card"><div class="label">{esc(item.label)}</div>'
                      f'<div class="value">{esc(item.value)}</div>{delta}</div>')  # fmt: skip
    return f'<div class="cards">{"".join(blocks)}</div>'


def images(figures: Mapping[str, Figure]) -> str:
    """Grille de graphiques, chacun avec sa légende."""
    blocks = [f'<figure><img alt="{esc(name)}" src="{figure_src(fig)}"></figure>' for name, fig in figures.items()]
    return f'<div class="grid">{"".join(blocks)}</div>'


def table(frame: pd.DataFrame, digits: int = 4, highlight: str | None = None, scroll: bool = False) -> str:
    """Tableau HTML : nombres alignés à droite et arrondis ; `highlight` = colonne booléenne à mettre en avant."""
    columns = [c for c in frame.columns if c != highlight]
    places = {c: 0 if _integral(frame[c]) else digits for c in columns}
    head = "".join(f'<th class="{_kind(frame[c])}">{esc(c)}</th>' for c in columns)
    rows = []
    for record in frame.to_dict("records"):
        selected = ' class="selected"' if highlight and record.get(highlight) else ""
        cells = "".join(f'<td class="{_kind(frame[c])}">{_format(record[c], places[c])}</td>' for c in columns)
        rows.append(f"<tr{selected}>{cells}</tr>")
    body = f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    return f'<div class="scroll">{body}</div>' if scroll else body


def markdown(text: str) -> str:
    """Markdown simple (titres, listes, tableaux, `code`, **gras**) : suffisant pour la fiche modèle générée."""
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set(":-") for c in cells):
                    rows.append(cells)
                i += 1
            head = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue
        if line.startswith("## "):
            out.append(f"<h3>{_inline(line[3:])}</h3>")
        elif line.startswith("# "):
            pass  # le titre de la fiche est déjà celui de la section
        elif line:
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return "".join(out)


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)


def _kind(series: pd.Series[Any]) -> str:
    return "num" if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) else ""


def _integral(series: pd.Series[Any]) -> bool:
    """Colonne flottante qui ne contient que des entiers (un classement avec des trous, par exemple)."""
    if not pd.api.types.is_float_dtype(series):
        return False
    values = series.dropna()
    return not values.empty and bool((values == values.round()).all())


def _format(value: object, digits: int) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '<span class="muted">—</span>'
    if isinstance(value, bool):
        return "oui" if value else ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return esc(value)
