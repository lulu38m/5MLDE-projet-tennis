"""`make urls` : chaque interface de la stack, son adresse, ses identifiants de démo, et si elle répond."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from tennis_mlops.config import Settings


@dataclass(frozen=True)
class Entry:
    """Une interface à ouvrir."""

    name: str
    url: str
    probe: str | None  # adresse de santé (None : rien à sonder)
    access: str  # identifiants, ou comment s'y connecter


def entries(settings: Settings) -> list[Entry]:
    """Les interfaces de la stack, dans l'ordre du parcours correcteur."""

    def u(value: object, path: str = "") -> str:
        return f"{str(value).rstrip('/')}{path}"

    db = urlsplit(settings.postgres_dsn.get_secret_value())
    grafana_password = settings.grafana_admin_password.get_secret_value()
    return [
        Entry("Interface de test", u(settings.ui_url), u(settings.ui_url, "/_stcore/health"), "sans compte"),
        Entry("API (docs)", u(settings.api_url, "/docs"), u(settings.api_url, "/health"), "sans compte"),
        Entry("Prefect", u(settings.prefect_ui_url), u(settings.prefect_api_url, "/health"), "sans compte"),
        Entry("MLflow", u(settings.mlflow_ui_url), u(settings.mlflow_tracking_uri, "/health"), "sans compte"),
        Entry(
            "Grafana",
            u(settings.grafana_ui_url, "/dashboards/f/tennis-mlops"),
            u(settings.grafana_ui_url, "/api/health"),
            f"lecture sans compte ; modifier : admin / {grafana_password}",
        ),
        Entry(
            "Prometheus", u(settings.prometheus_url, "/targets"), u(settings.prometheus_url, "/-/ready"), "sans compte"
        ),
        Entry(
            "Service d'alertes",
            u(settings.alert_sink_url, "/docs"),
            u(settings.alert_sink_url, "/health"),
            "sans compte (make alerts)",
        ),
        Entry(
            "mock_source (docs)",
            u(settings.mock_source_url, "/docs"),
            u(settings.mock_source_url, "/health"),
            "sans compte",
        ),
        Entry(
            "Great Expectations",
            "state/gx/gx/uncommitted/data_docs/local_site/index.html",
            None,
            "fichier local (make gx-docs)",
        ),
        Entry(
            "PostgreSQL",
            f"{db.hostname}:{db.port} base {db.path.lstrip('/')}",
            None,
            f"{db.username} / {db.password} (make db)",
        ),
    ]


def lines(settings: Settings, get: Callable[[str], int] | None = None) -> list[str]:
    """Le tableau de `make urls` ; ● répond, ○ ne répond pas (stack arrêtée ?)."""
    probe = get or _status
    rows = []
    for entry in entries(settings):
        mark = " " if entry.probe is None else ("●" if probe(entry.probe) == 200 else "○")
        rows.append(f"{mark} {entry.name:<20} {entry.url:<52} {entry.access}")
    return [*rows, "", "● répond · ○ ne répond pas (`uv run tennis up`, puis docker compose ps)"]


def _status(url: str) -> int:
    try:
        return httpx.get(url, timeout=2).status_code
    except httpx.HTTPError:
        return 0
