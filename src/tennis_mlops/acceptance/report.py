"""Résultat des tests d'acceptation : étapes vérifiées, statut par scénario, rapport HTML et JSON.

Le rapport (`state/acceptance/rapport.html`, ouvert par `make scenarios-report`) est autonome : un seul fichier,
sans script ni ressource externe. Chaque scénario y montre ce qui était attendu, chaque vérification faite (✅ / ❌)
et les liens vers les preuves (run Prefect, version MLflow).
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class Status(StrEnum):
    """Issue d'un scénario."""

    PASSED = "passed"  # toutes les vérifications ont réussi
    FAILED = "failed"  # une vérification a échoué : le comportement attendu n'est pas là
    ERROR = "error"  # le scénario n'a pas pu aller au bout (service injoignable, délai dépassé…)
    SKIPPED = "skipped"  # non lancé (sélection `ONLY=…`)


LABELS = {Status.PASSED: "✅ réussi", Status.FAILED: "❌ échoué", Status.ERROR: "⚠️ erreur", Status.SKIPPED: "— ignoré"}


@dataclass(frozen=True)
class Step:
    """Une vérification (ou une action) et son issue."""

    ok: bool
    text: str


@dataclass(frozen=True)
class Link:
    """Une preuve à ouvrir : run Prefect, version MLflow."""

    label: str
    url: str


class ScenarioFailedError(AssertionError):
    """Une vérification a échoué."""


@dataclass
class Recorder:
    """Ce que fait un scénario, étape par étape ; `expect` arrête le scénario à la première vérification fausse."""

    steps: list[Step] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)

    def step(self, text: str) -> None:
        """Une action faite (ou un constat), sans condition."""
        self.steps.append(Step(True, text))

    def expect(self, condition: object, text: str, failure: str | None = None) -> None:
        """Vérification : si elle est fausse, l'étape est notée ❌ et le scénario s'arrête."""
        if condition:
            self.steps.append(Step(True, text))
            return
        message = failure or text
        self.steps.append(Step(False, message))
        raise ScenarioFailedError(message)

    def link(self, label: str, url: str) -> None:
        """Une preuve à ouvrir."""
        self.links.append(Link(label, url))


@dataclass
class ScenarioResult:
    """Un scénario joué."""

    id: str
    title: str
    expected: str
    status: Status
    seconds: float = 0.0
    steps: list[Step] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    error: str = ""


@dataclass
class CampaignResult:
    """Tous les scénarios joués par un `make scenarios`."""

    started_at: datetime
    finished_at: datetime
    environment: dict[str, str]
    scenarios: list[ScenarioResult]
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Vrai si aucun scénario lancé n'a échoué."""
        played = [s for s in self.scenarios if s.status is not Status.SKIPPED]
        return bool(played) and all(s.status is Status.PASSED for s in played)

    def counts(self) -> dict[Status, int]:
        """Nombre de scénarios par issue."""
        return {status: sum(s.status is status for s in self.scenarios) for status in Status}

    def summary(self) -> str:
        """Une ligne pour le terminal."""
        c = self.counts()
        verdict = "VERT" if self.passed else "ROUGE"
        minutes = (self.finished_at - self.started_at).total_seconds() / 60
        return (f"{verdict} : {c[Status.PASSED]} réussi(s), {c[Status.FAILED]} échoué(s), {c[Status.ERROR]} en "
                f"erreur, {c[Status.SKIPPED]} ignoré(s) — {minutes:.1f} min")  # fmt: skip

    def to_json(self) -> str:
        """Le résultat complet, lisible par une machine."""
        data: dict[str, Any] = asdict(self)
        data["passed"] = self.passed
        return json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n"

    def write(self, directory: Path) -> Path:
        """Écrit `rapport.html` et `rapport.json` ; renvoie le chemin du HTML."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rapport.json").write_text(self.to_json(), encoding="utf-8")
        page = directory / "rapport.html"
        page.write_text(render_html(self), encoding="utf-8")
        return page


# --------------------------------------------------------------------------- HTML

STYLE = """
:root { --fg: #1d2330; --muted: #5d6675; --line: #e3e6ec; --bg: #ffffff; --soft: #f6f7f9;
        --ok: #1f7a43; --ko: #b42318; --warn: #a15c00; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e9ef; --muted: #9aa3b2; --line: #2c323d; --bg: #14171c; --soft: #1b1f26;
          --ok: #5cc98a; --ko: #f0837a; --warn: #e3a64a; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 0; }
.muted { color: var(--muted); }
.verdict { display: inline-block; margin: 16px 0 8px; padding: 6px 14px; border-radius: 999px; font-weight: 600; }
.verdict.ok { background: color-mix(in srgb, var(--ok) 15%, transparent); color: var(--ok); }
.verdict.ko { background: color-mix(in srgb, var(--ko) 15%, transparent); color: var(--ko); }
table { width: 100%; border-collapse: collapse; margin: 16px 0 32px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 13px; color: var(--muted); font-weight: 600; }
td.num { white-space: nowrap; text-align: right; font-variant-numeric: tabular-nums; }
td.passed, td.failed, td.error, td.skipped { white-space: nowrap; }
.passed { color: var(--ok); } .failed { color: var(--ko); } .error { color: var(--warn); }
.skipped { color: var(--muted); }
section { border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; margin: 16px 0;
          background: var(--soft); }
section header { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.expected { margin: 6px 0 10px; }
ul.steps { list-style: none; padding: 0; margin: 0; }
ul.steps li { padding: 3px 0 3px 26px; position: relative; }
ul.steps li::before { position: absolute; left: 0; }
ul.steps li.ok::before { content: "✅"; } ul.steps li.ko::before { content: "❌"; }
.links a { color: inherit; margin-right: 14px; }
pre { white-space: pre-wrap; word-break: break-word; font-size: 13px; margin: 8px 0 0; color: var(--ko); }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 16px; margin: 12px 0 0; font-size: 14px; }
dt { color: var(--muted); }
@media (max-width: 640px) { th:nth-child(3), td:nth-child(3) { display: none; } }
"""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def render_html(result: CampaignResult) -> str:
    """La page du rapport."""
    rows = "".join(
        f"<tr><td>{_e(s.id)}</td><td>{_e(s.title)}</td><td class='muted'>{_e(s.expected)}</td>"
        f"<td class='{s.status.value}'>{_e(LABELS[s.status])}</td><td class='num'>{s.seconds:.0f} s</td></tr>"
        for s in result.scenarios
    )
    sections = "".join(_section(s) for s in result.scenarios if s.status is not Status.SKIPPED)
    env = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in result.environment.items())
    notes = "".join(f"<li>{_e(n)}</li>" for n in result.notes)
    verdict = "ok" if result.passed else "ko"
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tests d'acceptation</title><style>{STYLE}</style></head>
<body><main>
<h1>Tests d'acceptation — tennis-mlops</h1>
<div class="muted">{_e(result.started_at.astimezone().strftime("%d/%m/%Y %H:%M"))} · <code>make scenarios</code></div>
<div class="verdict {verdict}">{_e(result.summary())}</div>
<dl>{env}</dl>
{f"<ul class='muted'>{notes}</ul>" if notes else ""}
<table><thead><tr><th>#</th><th>Scénario</th><th>Résultat attendu</th><th>Issue</th><th>Durée</th></tr></thead>
<tbody>{rows}</tbody></table>
{sections}
</main></body></html>
"""


def _section(s: ScenarioResult) -> str:
    steps = "".join(f"<li class='{'ok' if st.ok else 'ko'}'>{_e(st.text)}</li>" for st in s.steps)
    links = "".join(f"<a href='{_e(link.url)}'>{_e(link.label)}</a>" for link in s.links)
    error = f"<pre>{_e(s.error)}</pre>" if s.error else ""
    return f"""<section id="{_e(s.id)}"><header><h2>{_e(s.id)} · {_e(s.title)}</h2>
<span class="{s.status.value}">{_e(LABELS[s.status])} · {s.seconds:.0f} s</span></header>
<div class="expected muted">Attendu : {_e(s.expected)}</div>
<ul class="steps">{steps}</ul>{f"<div class='links'>{links}</div>" if links else ""}{error}</section>"""
