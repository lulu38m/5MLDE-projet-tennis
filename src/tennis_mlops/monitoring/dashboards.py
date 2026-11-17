"""Les trois dashboards Grafana, écrits en Python puis enregistrés en JSON dans `infra/grafana/dashboards/`.

`python -m tennis_mlops.monitoring.dashboards` (`make dashboards`) régénère les fichiers ; Grafana les charge au
démarrage (provisioning) et un test vérifie que les fichiers du dépôt sont à jour. Les panneaux sont écrits en code
plutôt qu'à la souris : chaque requête passe en revue de code, les trois dashboards partagent les mêmes conventions
(couleurs, unités, annotations), et rien ne se perd si le volume de Grafana est effacé.

- **Technique** (temps réel) : trafic, latences, erreurs, ressources des conteneurs, flows Prefect, modèle servi.
- **Métier** (temps **simulé** : l'axe est la date des matchs) : qualité en ligne, calibration, surprises,
  couverture du feedback, répartition des prédictions, dérive des entrées.
- **Croisé** : ce qui ne se lit qu'en rapprochant les deux : par version, par incident, par âge du modèle.

Sources : Prometheus (`uid: prometheus`) et la base du projet en lecture seule (`uid: tennis-pg`, compte
`grafana_reader`). Les annotations viennent de `pipeline_events`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tennis_mlops.training.drift import PSI_ALERT

PROM = {"type": "prometheus", "uid": "prometheus"}
PG = {"type": "grafana-postgresql-datasource", "uid": "tennis-pg"}
OUT = Path("infra/grafana/dashboards")
SCHEMA_VERSION = 41
SIMULATED_RANGE = {"from": "2015-01-01T00:00:00.000Z", "to": "2026-06-01T00:00:00.000Z"}
GREEN, ORANGE, RED, BLUE = "green", "orange", "red", "blue"
Json = dict[str, Any]

# ====================================================================== requêtes PromQL

API = 'job="api"'
RATE = "$__rate_interval"
REQUESTS = f"tennis_api_requests_total{{{API}}}"
REQ_RATE = f"sum(rate({REQUESTS}[{RATE}]))"
# `or vector(0)` : tant qu'aucune erreur n'est arrivée, la série n'existe pas ; on affiche 0 plutôt que « No data ».
ERR_5XX = f'(sum(rate(tennis_api_requests_total{{{API},status=~"5.."}}[{RATE}])) or vector(0))'
ERR_4_5XX = f'(sum(rate(tennis_api_requests_total{{{API},status=~"[45].."}}[{RATE}])) or vector(0))'
NON_2XX = f'sum by (status) (rate(tennis_api_requests_total{{{API},status!~"2.."}}[{RATE}]))'
BUCKETS = f"tennis_api_request_duration_seconds_bucket{{{API}}}"
PREDICT_BUCKETS = f'tennis_api_request_duration_seconds_bucket{{{API},path=~"/predict.*"}}'
CPU = f"sum by (service) (rate(tennis_container_cpu_seconds_total[{RATE}]))"
MEMORY = "sum by (service) (tennis_container_memory_bytes)"
MEMORY_SHARE = "max by (service) (tennis_container_memory_bytes / tennis_container_memory_limit_bytes)"
API_CPU_15M = 'sum(rate(tennis_container_cpu_seconds_total{service="api"}[15m]))'
CORRECT_15M = f'sum(rate(tennis_api_feedback_predictions_total{{{API},correct="true"}}[15m]))'


def quantile(q: float, by: str, buckets: str) -> str:
    """Quantile estimé depuis l'histogramme de latence."""
    return f"histogram_quantile({q}, sum by ({by}) (rate({buckets}[{RATE}])))"


# Sans trafic, le quantile vaut NaN : on n'affiche alors rien (« pas de trafic ») plutôt que « NaN ».
PREDICT_P95 = f"{quantile(0.95, 'le', PREDICT_BUCKETS)} and sum(rate({PREDICT_BUCKETS}[{RATE}])) > 0"


# ====================================================================== requêtes SQL

# Les prédictions du scoring (faites avant le match) dont le résultat validé est arrivé, sur la période affichée.
# `$__timeFilter` porte sur la date du match : le dashboard Métier est en temps simulé.
JUDGED = """
FROM predictions p JOIN feedbacks f ON f.match_id = p.match_id
WHERE p.client = 'scoring' AND $__timeFilter(p.match_date::timestamp)
"""
SCORED = "FROM predictions p WHERE p.client = 'scoring' AND $__timeFilter(p.match_date::timestamp)"
CORRECT = "CASE WHEN f.winner_id = p.predicted_winner_id THEN 1.0 ELSE 0 END"
WRONG = "CASE WHEN f.winner_id <> p.predicted_winner_id THEN 1.0 ELSE 0 END"
WINNER_P = "CASE WHEN f.winner_id = p.p1_id THEN p.p1_win_probability ELSE 1 - p.p1_win_probability END"
FAVOURITE_P = "GREATEST(p.p1_win_probability, 1 - p.p1_win_probability)"
LOG_LOSS = f"-LN(GREATEST({WINNER_P}, 1e-15))"
BRIER = f"POWER(1 - {WINNER_P}, 2)"
WEEK = "date_trunc('week', p.match_date)::timestamp"
MONTH = "date_trunc('month', p.match_date)::timestamp"

Q_ACCURACY = f"SELECT AVG({CORRECT}) AS accuracy {JUDGED}"
Q_LOG_LOSS = f'SELECT AVG({LOG_LOSS}) AS "log loss" {JUDGED}'
Q_BRIER = f"SELECT AVG({BRIER}) AS brier {JUDGED}"
Q_JUDGED = f"SELECT COUNT(*) AS jugés {JUDGED}"
Q_COVERAGE = """
SELECT AVG(CASE WHEN f.match_id IS NULL THEN 0.0 ELSE 1 END) AS couverture
FROM predictions p LEFT JOIN feedbacks f ON f.match_id = p.match_id
WHERE p.client = 'scoring' AND $__timeFilter(p.match_date::timestamp)
  AND p.match_date <= (SELECT MAX(match_date) FROM matches)
"""
Q_UPSETS = f"SELECT AVG({WRONG}) AS surprises {JUDGED} AND {FAVOURITE_P} >= 0.7"
Q_WEEKLY = f"""
SELECT {WEEK} AS time, AVG({CORRECT}) AS accuracy, AVG({LOG_LOSS}) AS "log loss"
{JUDGED} GROUP BY 1 ORDER BY 1
"""
Q_WEEKLY_BRIER = f"SELECT {WEEK} AS time, AVG({BRIER}) AS brier {JUDGED} GROUP BY 1 ORDER BY 1"
Q_CALIBRATION = f"""
SELECT LPAD((FLOOR({FAVOURITE_P} * 10) * 10)::int::text, 2, '0') || '-'
       || (FLOOR({FAVOURITE_P} * 10) * 10 + 10)::int::text || ' %' AS tranche,
       AVG({FAVOURITE_P}) AS annoncée, AVG({CORRECT}) AS réelle, COUNT(*) AS matchs
{JUDGED} GROUP BY 1 ORDER BY 1
"""
Q_UPSETS_MONTHLY = f'SELECT {MONTH} AS time, AVG({WRONG}) AS "favori battu" {JUDGED} GROUP BY 1 ORDER BY 1'
Q_WEEKLY_COVERAGE = f"""
SELECT {WEEK} AS time, COUNT(*) AS prédits, COUNT(f.match_id) AS "avec résultat"
FROM predictions p LEFT JOIN feedbacks f ON f.match_id = p.match_id
WHERE p.client = 'scoring' AND $__timeFilter(p.match_date::timestamp)
GROUP BY 1 ORDER BY 1
"""
Q_BY_SURFACE = f"SELECT COALESCE(p.surface, 'inconnue') AS surface, COUNT(*) AS prédictions {SCORED} GROUP BY 1"
LEVELS = """
CASE p.tourney_level WHEN 'G' THEN 'Grand Chelem' WHEN 'M' THEN 'Masters 1000' WHEN 'A' THEN 'ATP 250 / 500'
     WHEN 'F' THEN 'Masters de fin d''année' WHEN 'D' THEN 'Coupe Davis' WHEN 'O' THEN 'Jeux olympiques'
     ELSE 'autre' END
"""
Q_BY_LEVEL = f"SELECT {LEVELS} AS niveau, COUNT(*) AS prédictions {SCORED} GROUP BY 1"
Q_SURFACE_QUALITY = f"""
SELECT COALESCE(p.surface, 'inconnue') AS surface, AVG({CORRECT}) AS accuracy, AVG({LOG_LOSS}) AS "log loss"
{JUDGED} GROUP BY 1 ORDER BY 1
"""
Q_TOURNAMENTS = """
SELECT COALESCE(p.tourney_name, split_part(p.match_id, '-', 1) || '-' || split_part(p.match_id, '-', 2)) AS tournoi,
       MIN(p.match_date) AS début, COUNT(*) AS prédits, COUNT(f.match_id) AS jugés,
       AVG(CASE WHEN f.winner_id IS NULL THEN NULL WHEN f.winner_id = p.predicted_winner_id THEN 1.0 ELSE 0 END)
         AS accuracy
FROM predictions p LEFT JOIN feedbacks f ON f.match_id = p.match_id
WHERE p.client = 'scoring' AND $__timeFilter(p.match_date::timestamp)
GROUP BY 1 ORDER BY 3 DESC, 2 DESC LIMIT 15
"""

# Croisé (temps réel)
VERSION_ORDER = "LPAD(model_version, 6, '0')"
Q_VERSIONS = f"""
SELECT 'v' || model_version AS version, predictions AS prédictions, with_feedback AS jugées, accuracy,
       log_loss AS "log loss", brier, mean_request_ms AS "latence moyenne"
FROM online_metrics ORDER BY {VERSION_ORDER}
"""
Q_VERSIONS_BARS = f"""
SELECT 'v' || model_version AS version, accuracy, mean_request_ms AS "latence (ms)"
FROM online_metrics WHERE with_feedback > 0 ORDER BY {VERSION_ORDER}
"""
Q_UNPREDICTED = """
SELECT date_trunc('minute', m.ingested_at) AS time,
       COUNT(*) FILTER (WHERE p.match_id IS NULL) AS "sans prédiction",
       COUNT(*) FILTER (WHERE p.match_id IS NOT NULL) AS "prédits avant"
FROM matches m
LEFT JOIN (SELECT DISTINCT match_id FROM predictions WHERE client = 'scoring') p ON p.match_id = m.match_id
WHERE $__timeFilter(m.ingested_at)
  AND m.match_date >= (SELECT MIN(match_date) FROM predictions WHERE client = 'scoring')
GROUP BY 1 ORDER BY 1
"""
Q_MODEL_AGE = f"""
SELECT LPAD((FLOOR((p.match_date - p.model_data_end) / 30.0) * 30)::int::text, 4, '0') || ' j' AS "âge du modèle",
       AVG({CORRECT}) AS accuracy, COUNT(*) AS matchs
FROM predictions p JOIN feedbacks f ON f.match_id = p.match_id
WHERE p.client = 'scoring' AND p.model_data_end IS NOT NULL AND p.match_date > p.model_data_end
GROUP BY 1 ORDER BY 1
"""
Q_BATCH = """
SELECT CASE WHEN predictions <= 1 THEN '1' WHEN predictions <= 10 THEN '2-10' WHEN predictions <= 50 THEN '11-50'
            WHEN predictions <= 100 THEN '51-100' ELSE '101+' END AS "matchs par appel",
       AVG(duration_ms) AS "latence moyenne", AVG(duration_ms / NULLIF(predictions, 0)) AS "par match",
       COUNT(*) AS appels
FROM api_requests
WHERE path IN ('/predict', '/predict/batch') AND status_code = 200 AND predictions > 0
  AND $__timeFilter(received_at)
GROUP BY 1 ORDER BY MIN(predictions)
"""


def prom(expr: str, legend: str = "", ref: str = "A", instant: bool = False) -> Json:
    """Une requête PromQL."""
    return {
        "refId": ref,
        "datasource": PROM,
        "expr": expr,
        "legendFormat": legend or "__auto",
        "range": not instant,
        "instant": instant,
        "editorMode": "code",
    }


def sql(raw: str, ref: str = "A", table: bool = False) -> Json:
    """Une requête SQL (série temporelle : une colonne `time`, puis les valeurs ; sinon une table)."""
    return {
        "refId": ref,
        "datasource": PG,
        "rawSql": " ".join(raw.split()),
        "format": "table" if table else "time_series",
        "rawQuery": True,
        "editorMode": "code",
    }


# ====================================================================== panneaux


def steps(*values: tuple[float | None, str]) -> Json:
    """Seuils de couleur."""
    return {"mode": "absolute", "steps": [{"value": v, "color": c} for v, c in values]}


def value_map(values: dict[str, tuple[str, str]]) -> list[Json]:
    """Valeurs remplacées par un libellé et une couleur."""
    options = {k: {"text": text, "color": color, "index": i} for i, (k, (text, color)) in enumerate(values.items())}
    return [{"type": "value", "options": options}]


def override(series: str, unit: str | None = None, color: str | None = None, right: bool = False) -> Json:
    """Unité, couleur fixe ou axe à droite d'une série."""
    properties: list[Json] = []
    if unit:
        properties.append({"id": "unit", "value": unit})
    if color:
        properties.append({"id": "color", "value": {"mode": "fixed", "fixedColor": color}})
    if right:
        properties.append({"id": "custom.axisPlacement", "value": "right"})
    return {"matcher": {"id": "byName", "options": series}, "properties": properties}


def panel(kind: str, title: str, targets: list[Json], description: str = "", **config: Any) -> Json:  # noqa: ANN401
    """Un panneau Grafana ; sa position est fixée par `Board.add`.

    `config` : `unit`, `options`, `defaults` (fieldConfig), `overrides`, et toute autre clé du panneau.
    """
    defaults: Json = {"unit": config.pop("unit")} if config.get("unit") else {}
    config.pop("unit", None)
    defaults |= config.pop("defaults", {})
    return {
        "type": kind,
        "title": title,
        "description": description,
        "datasource": targets[0]["datasource"],
        "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": config.pop("overrides", [])},
        "options": config.pop("options", {}),
    } | config


LAST = {"calcs": ["lastNotNull"], "fields": "", "values": False}


def timeseries(title: str, targets: list[Json], description: str = "", *, unit: str | None = None,
               stack: bool = False, bars: bool = False, points: bool = False, minimum: float | None = None,
               overrides: list[Json] | None = None) -> Json:  # fmt: skip
    """Courbes (ou barres) dans le temps ; légende en tableau avec moyenne et dernière valeur."""
    custom = {
        "drawStyle": "bars" if bars else "line",
        "lineWidth": 2,
        "fillOpacity": 70 if bars else 10,
        "showPoints": "always" if points else "never",
        "pointSize": 6,
        "spanNulls": True,
        "stacking": {"mode": "normal" if stack else "none"},
    }
    defaults: Json = {"custom": custom} | ({"min": minimum} if minimum is not None else {})
    options = {
        "legend": {"displayMode": "table", "placement": "bottom", "calcs": ["mean", "lastNotNull"]},
        "tooltip": {"mode": "multi", "sort": "desc"},
    }
    return panel("timeseries", title, targets, description, unit=unit, defaults=defaults, options=options,
                 overrides=overrides or [])  # fmt: skip


def stat(title: str, targets: list[Json], description: str = "", *, unit: str | None = None,
         thresholds: Json | None = None, mappings: list[Json] | None = None, decimals: int | None = None,
         text_mode: str = "value", color_mode: str = "background", no_value: str | None = None) -> Json:  # fmt: skip
    """Un chiffre, coloré par seuils."""
    defaults: Json = {"thresholds": thresholds or steps((None, BLUE)), "color": {"mode": "thresholds"}}
    defaults |= {"noValue": no_value} if no_value else {}
    defaults |= {"mappings": mappings} if mappings else {}
    defaults |= {"decimals": decimals} if decimals is not None else {}
    options = {"reduceOptions": LAST, "colorMode": color_mode, "graphMode": "none", "textMode": text_mode,
               "justifyMode": "center", "wideLayout": True}  # fmt: skip
    return panel("stat", title, targets, description, unit=unit, defaults=defaults, options=options)


def bargauge(title: str, targets: list[Json], description: str = "", *, unit: str | None = None,
             thresholds: Json | None = None, maximum: float | None = None) -> Json:  # fmt: skip
    """Barres horizontales, une par série (dernière valeur)."""
    defaults: Json = {"thresholds": thresholds or steps((None, BLUE)), "color": {"mode": "thresholds"}, "min": 0}
    defaults |= {"max": maximum} if maximum is not None else {}
    options = {"reduceOptions": LAST, "orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True,
               "valueMode": "color", "namePlacement": "left"}  # fmt: skip
    return panel("bargauge", title, targets, description, unit=unit, defaults=defaults, options=options)


def barchart(title: str, targets: list[Json], description: str = "", *, x_field: str, unit: str | None = None,
             overrides: list[Json] | None = None) -> Json:  # fmt: skip
    """Barres verticales groupées sur une table : une colonne texte en abscisse, une barre par colonne numérique."""
    options = {
        "xField": x_field,
        "orientation": "vertical",
        "showValue": "auto",
        "stacking": "none",
        "groupWidth": 0.7,
        "barWidth": 0.9,
        "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
        "tooltip": {"mode": "multi"},
    }
    defaults = {"color": {"mode": "palette-classic"}}
    return panel("barchart", title, targets, description, unit=unit, defaults=defaults, options=options,
                 overrides=overrides or [])  # fmt: skip


def table(title: str, targets: list[Json], description: str = "", *, overrides: list[Json] | None = None) -> Json:
    """Un tableau."""
    options = {"showHeader": True, "cellHeight": "sm"}
    return panel("table", title, targets, description, options=options, overrides=overrides or [])


def timeline(title: str, targets: list[Json], description: str = "", *, mappings: list[Json],
             thresholds: Json) -> Json:  # fmt: skip
    """Bandes d'état dans le temps."""
    defaults = {"thresholds": thresholds, "color": {"mode": "thresholds"}, "mappings": mappings}
    options = {"showValue": "never", "mergeValues": True, "rowHeight": 0.8, "legend": {"showLegend": False}}
    return panel("state-timeline", title, targets, description, defaults=defaults, options=options)


def piechart(title: str, targets: list[Json], description: str = "") -> Json:
    """Un anneau : la part de chaque ligne."""
    options = {
        "pieType": "donut",
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": True},
        "legend": {"displayMode": "table", "placement": "right", "values": ["value", "percent"], "showLegend": True},
    }
    return panel("piechart", title, targets, description, defaults={"color": {"mode": "palette-classic"}},
                 options=options)  # fmt: skip


def text(title: str, content: str) -> Json:
    """Un panneau de texte (Markdown)."""
    return {"type": "text", "title": title, "options": {"mode": "markdown", "content": content}}


# ====================================================================== annotations (pipeline_events)

# Un événement par pas de simulation : trop pour des annotations (ils restent dans le journal et sur la page Modèle).
NOISY = "('ingestion_completed', 'alert')"
ANNOTATION_GROUPS = (
    ("Incidents", "failure", "red"),
    ("Dégradations", "warning", "orange"),
    ("Mises en service", "success", "green"),
    ("Autres événements", "info", "blue"),
)
# Sur un axe simulé, un événement sans date simulée (une promotion) prend celle de la dernière ingestion qui le
# précède : la date à laquelle en était l'horloge quand il est arrivé.
SIMULATED_MOMENT = """
COALESCE(e.simulated_date, (SELECT MAX(i.simulated_date) FROM pipeline_events i
                            WHERE i.simulated_date IS NOT NULL AND i.occurred_at <= e.occurred_at))::timestamp
"""
Q_ANNOTATIONS = """
SELECT time, text, tags FROM (
  SELECT {moment} AS time, e.title || COALESCE(' — ' || NULLIF(LEFT(e.detail, 300), ''), '') AS text, e.kind AS tags
  FROM pipeline_events e WHERE e.severity = '{severity}' AND e.kind NOT IN {noisy}
) a WHERE time IS NOT NULL AND $__timeFilter(time) ORDER BY time
"""


def event_annotations(simulated: bool) -> list[Json]:
    """Incidents en rouge, dégradations en orange, mises en service et rollbacks en vert, le reste en bleu."""
    moment = SIMULATED_MOMENT if simulated else "e.occurred_at"
    out: list[Json] = [{
        "builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"}, "enable": True, "hide": True,
        "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts", "type": "dashboard",
    }]  # fmt: skip
    for name, severity, color in ANNOTATION_GROUPS:
        query = Q_ANNOTATIONS.format(moment=moment, severity=severity, noisy=NOISY)
        out.append({"name": name, "datasource": PG, "enable": True, "iconColor": color,
                    "target": sql(query, ref="Anno", table=True)})  # fmt: skip
    return out


# ====================================================================== disposition


@dataclass
class Board:
    """Un dashboard : panneaux placés de gauche à droite, rangée par rangée, sur 24 colonnes."""

    uid: str
    title: str
    description: str
    time: Json
    refresh: str
    simulated: bool
    panels: list[Json] = field(default_factory=list)
    _x: int = 0
    _y: int = 0
    _height: int = 0

    def row(self, title: str) -> None:
        """Une rangée titrée, sur toute la largeur."""
        self._newline()
        self.panels.append({"type": "row", "title": title, "collapsed": False, "panels": [],
                            "gridPos": {"h": 1, "w": 24, "x": 0, "y": self._y}})  # fmt: skip
        self._y += 1

    def add(self, item: Json, w: int = 12, h: int = 8) -> None:
        """Place un panneau ; passe à la ligne s'il ne tient plus."""
        if self._x + w > 24:
            self._newline()
        self.panels.append(item | {"gridPos": {"h": h, "w": w, "x": self._x, "y": self._y}})
        self._x += w
        self._height = max(self._height, h)

    def _newline(self) -> None:
        if self._x:
            self._y += self._height
        self._x, self._height = 0, 0

    def json(self) -> Json:
        """Le modèle JSON de Grafana."""
        links = [{"title": "Dashboards tennis-mlops", "type": "dashboards", "tags": ["tennis-mlops"],
                  "asDropdown": False, "includeVars": False, "keepTime": False, "targetBlank": False}]  # fmt: skip
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": ["tennis-mlops"],
            "timezone": "browser",
            "schemaVersion": SCHEMA_VERSION,
            "version": 1,
            "editable": True,
            "graphTooltip": 1,
            "refresh": self.refresh,
            "time": self.time,
            "timepicker": {},
            "templating": {"list": []},
            "links": links,
            "annotations": {"list": event_annotations(self.simulated)},
            "panels": [p | {"id": i + 1} for i, p in enumerate(self.panels)],
        }


# ====================================================================== technique


def technique() -> Board:
    """Le service tient-il ? Trafic, latences, erreurs, ressources, flows, modèle."""
    b = Board("tennis-technique", "Tennis · Technique",
              "Temps réel : trafic et latences de l'API, erreurs, CPU et mémoire des conteneurs, flows Prefect.",
              {"from": "now-1h", "to": "now"}, "10s", simulated=False)  # fmt: skip
    ready = value_map({"0": ("pas prêt", RED), "1": ("dégradé", ORANGE), "2": ("prêt", GREEN)})
    up = value_map({"0": ("arrêté", RED), "1": ("en ligne", GREEN)})

    b.row("État")
    b.add(stat("API", [prom(f"tennis_api_ready{{{API}}}", "API")],
               "Prêt : modèle et historique chargés. Dégradé : MLflow ou la base injoignable, le cache sert.",
               thresholds=steps((None, RED), (1, ORANGE), (2, GREEN)), mappings=ready), 4, 4)  # fmt: skip
    b.add(stat("Version servie", [prom(f"tennis_api_model_info{{{API}}} == 1", "v{{version}}", instant=True)],
               "Alias production lu par l'API (relu toutes les 30 s)", text_mode="name", color_mode="none"),
          4, 4)  # fmt: skip
    b.add(stat("Requêtes / s", [prom(REQ_RATE, "req/s")], unit="reqps", decimals=2), 4, 4)
    b.add(stat("Latence p95 des prédictions", [prom(PREDICT_P95, "p95")],
               "POST /predict et /predict/batch", unit="s", no_value="pas de trafic",
               thresholds=steps((None, GREEN), (0.2, ORANGE), (0.5, RED))), 4, 4)  # fmt: skip
    b.add(stat("Erreurs serveur", [prom(f"{ERR_5XX} / clamp_min({REQ_RATE}, 1e-9)", "5xx")],
               "Part des réponses 5xx", unit="percentunit", decimals=1,
               thresholds=steps((None, GREEN), (0.01, ORANGE), (0.05, RED))), 4, 4)  # fmt: skip
    b.add(stat("Horloge simulée", [prom("tennis_simulated_clock_timestamp * 1000", "date")],
               "Date de la source (mock_source)", unit="dateTimeAsIsoNoDateIfToday", color_mode="none"),
          4, 4)  # fmt: skip
    b.add(timeline("Disponibilité des services", [prom("tennis_service_up", "{{service}}")],
                   "Sonde de santé de chaque service, toutes les 15 s (exporteur)", mappings=up,
                   thresholds=steps((None, RED), (1, GREEN))), 24, 6)  # fmt: skip

    b.row("API")
    b.add(timeseries("Requêtes par endpoint", [prom(f"sum by (path) (rate({REQUESTS}[{RATE}]))", "{{path}}")],
                     unit="reqps", stack=True))  # fmt: skip
    latency = [prom(quantile(q, "le, path", BUCKETS), f"{{{{path}}}} p{round(q * 100)}", ref)
               for q, ref in ((0.5, "A"), (0.95, "B"), (0.99, "C"))]  # fmt: skip
    b.add(timeseries("Latence par endpoint (p50 · p95 · p99)", latency, "Quantiles estimés sur l'histogramme",
                     unit="s"))  # fmt: skip
    b.add(timeseries("Réponses hors 2xx", [prom(NON_2XX, "{{status}}")], "Par code HTTP", unit="reqps", stack=True,
                     bars=True), h=7)  # fmt: skip
    b.add(timeseries("Erreurs par type (sur 5 min)",
                     [prom(f"sum by (type) (increase(tennis_api_errors_total{{{API}}}[5m]))", "{{type}}")],
                     "validation, invalid_match, unavailable, internal…", unit="short", stack=True, bars=True),
          h=7)  # fmt: skip
    b.add(timeseries("Matchs prédits par client",
                     [prom(f"sum by (client) (rate(tennis_api_predictions_total{{{API}}}[{RATE}]))", "{{client}}")],
                     "scoring, ui, tests de charge…", unit="short", stack=True), h=7)  # fmt: skip
    writer = [prom(f"tennis_api_writer_pending{{{API}}}", "en attente", "A"),
              prom(f"tennis_api_writer_dropped{{{API}}}", "perdues", "B"),
              prom(f"tennis_api_writer_failed{{{API}}}", "refusées par la base", "C")]  # fmt: skip
    b.add(timeseries("File d'écriture des traces", writer, "Perdues ou refusées au-dessus de 0 : la base ne suit pas",
                     unit="short", overrides=[override("perdues", color=RED),
                                              override("refusées par la base", color=ORANGE)]), h=7)  # fmt: skip

    b.row("Conteneurs (API Docker, comme docker stats)")
    legend = "{{service}}"
    b.add(timeseries("CPU par service", [prom(CPU, legend)], "Cœurs utilisés", unit="short", stack=True), 9)
    b.add(timeseries("Mémoire par service", [prom(MEMORY, legend)], "Mémoire de travail", unit="bytes", stack=True), 9)
    b.add(bargauge("Mémoire / limite", [prom(MEMORY_SHARE, legend, instant=True)],
                   "Part de la limite deploy.resources.limits.memory : au-delà de 90 %, le conteneur risque l'OOM",
                   unit="percentunit", thresholds=steps((None, GREEN), (0.75, ORANGE), (0.9, RED)), maximum=1),
          6)  # fmt: skip

    b.row("Flows Prefect et modèle")
    b.add(bargauge("Runs des dernières 24 h",
                   [prom("sum by (flow, state) (tennis_prefect_flow_runs)", "{{flow}} · {{state}}", instant=True)],
                   "Par flow et par état (Completed, Failed, Crashed…)", unit="short"), 12, 8)  # fmt: skip
    b.add(bargauge("Durée du dernier run", [prom("tennis_prefect_flow_run_seconds", "{{flow}}", instant=True)],
                   "Run terminé le plus récent de chaque flow", unit="s"), 6, 8)  # fmt: skip
    b.add(bargauge("Relances (24 h)", [prom("tennis_prefect_retries", "{{flow}}", instant=True)],
                   "Tasks et flows relancés après un échec (retries Prefect)", unit="short",
                   thresholds=steps((None, GREEN), (1, ORANGE), (10, RED))), 6, 8)  # fmt: skip
    loads = [prom(f"tennis_api_model_load_seconds{{{API}}}", "chargement du modèle", "A"),
             prom(f"tennis_api_features_build_seconds{{{API}}}", "rejeu de l'historique", "B")]  # fmt: skip
    b.add(timeseries("Chargements de l'API", loads,
                     "Durée du dernier chargement : à chaque nouvelle version, à chaque nouveau lot", unit="s"),
          h=7)  # fmt: skip
    registry = [prom(f"tennis_api_model_stale{{{API}}}", "registre injoignable", "A"),
                prom(f"tennis_api_model_reloads{{{API}}}", "changements de version", "B")]  # fmt: skip
    b.add(timeseries("Registre MLflow", registry, "Injoignable (1) : l'API sert la version en cache", unit="short",
                     minimum=0, overrides=[override("registre injoignable", color=RED)]), h=7)  # fmt: skip
    return b


# ====================================================================== métier


def metier() -> Board:
    """Le modèle prédit-il bien ? L'axe des temps est la date des matchs (temps simulé)."""
    b = Board("tennis-metier", "Tennis · Métier",
              "Temps simulé (l'axe est la date des matchs) : prédictions du scoring faites avant chaque match, jugées "
              "sur le résultat validé ; calibration, surprises, couverture, dérive des entrées.",
              SIMULATED_RANGE, "30s", simulated=True)  # fmt: skip
    b.row("Qualité en ligne sur la période affichée")
    b.add(stat("Accuracy", [sql(Q_ACCURACY, table=True)], "Le favori du modèle a gagné", unit="percentunit",
               decimals=1, thresholds=steps((None, RED), (0.6, ORANGE), (0.65, GREEN))), 4, 4)  # fmt: skip
    b.add(stat("Log loss", [sql(Q_LOG_LOSS, table=True)], "Plus bas = meilleur ; 0,693 = pile ou face", decimals=3,
               thresholds=steps((None, GREEN), (0.62, ORANGE), (0.66, RED))), 4, 4)  # fmt: skip
    b.add(stat("Brier", [sql(Q_BRIER, table=True)], "Erreur quadratique sur la probabilité ; 0,25 = pile ou face",
               decimals=3, thresholds=steps((None, GREEN), (0.22, ORANGE), (0.24, RED))), 4, 4)  # fmt: skip
    b.add(stat("Matchs jugés", [sql(Q_JUDGED, table=True)], "Prédits avant le match, résultat validé", unit="short",
               color_mode="none"), 4, 4)  # fmt: skip
    b.add(stat("Couverture du feedback", [sql(Q_COVERAGE, table=True)],
               "Prédictions dont le résultat est arrivé, parmi celles dont la date est passée", unit="percentunit",
               decimals=1, thresholds=steps((None, RED), (0.8, ORANGE), (0.95, GREEN))), 4, 4)  # fmt: skip
    b.add(stat("Favoris à 70 % battus", [sql(Q_UPSETS, table=True)],
               "Un modèle calibré en perd moins de 30 %", unit="percentunit", decimals=1,
               thresholds=steps((None, GREEN), (0.25, ORANGE), (0.3, RED))), 4, 4)  # fmt: skip
    quality = [override("accuracy", unit="percentunit", color=GREEN), override("log loss", color=ORANGE, right=True)]
    b.add(timeseries("Accuracy et log loss par semaine", [sql(Q_WEEKLY)],
                     "Une valeur par semaine de matchs ; les annotations marquent mises en service et incidents",
                     points=True, overrides=quality), 16, 9)  # fmt: skip
    b.add(timeseries("Brier par semaine", [sql(Q_WEEKLY_BRIER)], "0,25 = pile ou face", points=True), 8, 9)

    b.row("Calibration et surprises")
    hidden = {"id": "custom.hideFrom", "value": {"viz": True, "legend": True, "tooltip": False}}
    b.add(barchart("Calibration du favori", [sql(Q_CALIBRATION, table=True)],
                   "Probabilité annoncée pour le favori et part de victoires réelle, par tranche : un modèle bien "
                   "calibré a ses deux barres à la même hauteur", x_field="tranche", unit="percentunit",
                   overrides=[{"matcher": {"id": "byName", "options": "matchs"},
                               "properties": [{"id": "unit", "value": "short"}, hidden]}]), 14, 9)  # fmt: skip
    b.add(timeseries("Favoris battus par mois", [sql(Q_UPSETS_MONTHLY)],
                     "Part des matchs où le modèle s'est trompé de vainqueur", unit="percentunit", bars=True,
                     minimum=0), 10, 9)  # fmt: skip

    b.row("Ce qu'on prédit")
    b.add(timeseries("Prédits et jugés par semaine", [sql(Q_WEEKLY_COVERAGE)],
                     "L'écart : matchs prédits dont le résultat n'est pas (encore) validé", unit="short", bars=True,
                     overrides=[override("avec résultat", color=GREEN), override("prédits", color=BLUE)]))  # fmt: skip
    b.add(piechart("Par surface", [sql(Q_BY_SURFACE, table=True)]), 6, 8)
    b.add(piechart("Par niveau de tournoi", [sql(Q_BY_LEVEL, table=True)]), 6, 8)
    b.add(barchart("Qualité par surface", [sql(Q_SURFACE_QUALITY, table=True)],
                   "Accuracy et log loss des prédictions jugées, sur la période affichée", x_field="surface",
                   overrides=[override("accuracy", unit="percentunit"), override("log loss", unit="none")]),
          )  # fmt: skip
    b.add(table("Tournois les plus prédits", [sql(Q_TOURNAMENTS, table=True)],
                "Prédits avant le match, et combien sont déjà jugés (un tournoi en cours n'a pas encore de résultat)",
                overrides=[override("accuracy", unit="percentunit")]))  # fmt: skip

    b.row("Dérive des entrées (temps réel : les 500 dernières prédictions contre l'entraînement du modèle servi)")
    b.add(bargauge("PSI par feature", [prom(f"tennis_api_feature_drift_psi{{{API}}}", "{{feature}}", instant=True)],
                   "Moins de 0,1 : stable. De 0,1 à 0,2 : à surveiller. Au-delà : le modèle extrapole.",
                   thresholds=steps((None, GREEN), (0.1, ORANGE), (PSI_ALERT, RED)), maximum=0.5)
          | {"timeFrom": "now-15m"}, 16, 9)  # fmt: skip
    b.add(text("Lire la dérive", DRIFT_HELP), 8, 9)
    return b


DRIFT_HELP = """\
Le **PSI** compare la distribution de chaque feature sur les 500 dernières prédictions servies à celle des deux
années d'entraînement du modèle en production (jusqu'à son tag `data_window_end`).

On compare l'**écart** entre les deux joueurs, en valeur absolue : l'ordre joueur 1 / joueur 2 n'est pas le même à
l'entraînement et dans un tableau.

Le panneau est en temps réel (dernier quart d'heure), quel que soit l'intervalle simulé choisi en haut.
"""


# ====================================================================== croisé


def croise() -> Board:
    """Ce qui ne se lit qu'en croisant technique et métier."""
    b = Board("tennis-croise", "Tennis · Croisé",
              "Par version de modèle, par incident, par âge du modèle : le coût et la qualité ensemble.",
              {"from": "now-6h", "to": "now"}, "30s", simulated=False)  # fmt: skip
    b.row("Le nouveau modèle coûte-t-il plus cher ?")
    b.add(table("Qualité et latence par version", [sql(Q_VERSIONS, table=True)],
                "Depuis le début (vue online_metrics) : une ligne par version servie",
                overrides=[override("accuracy", unit="percentunit"), override("latence moyenne", unit="ms")]),
          )  # fmt: skip
    b.add(barchart("Accuracy et latence par version", [sql(Q_VERSIONS_BARS, table=True)],
                   "Une version plus juste mais plus lente se voit tout de suite", x_field="version",
                   overrides=[override("accuracy", unit="percentunit"),
                              override("latence (ms)", unit="ms", right=True)]))  # fmt: skip

    b.row("Que font les incidents aux erreurs et aux prédictions ?")
    b.add(timeline("Incident en cours sur la source", [prom('tennis_source_chaos{mode!="off"}', "{{mode}}")],
                   "make chaos MODE=503 · slow · corrupt · schema", mappings=value_map({"0": ("normal", GREEN),
                                                                                       "1": ("en cours", RED)}),
                   thresholds=steps((None, GREEN), (1, RED))), 24, 5)  # fmt: skip
    signals = [prom(ERR_4_5XX, "erreurs API (req/s)", "A"),
               prom("count(tennis_service_up == 0) or vector(0)", "services injoignables", "B")]  # fmt: skip
    b.add(timeseries("Erreurs de l'API et services injoignables", signals,
                     "Les annotations rouges et orange sont les incidents du journal (lot refusé, ingestion dégradée)",
                     unit="short", overrides=[override("erreurs API (req/s)", color=RED),
                                              override("services injoignables", color=ORANGE)]))  # fmt: skip
    b.add(timeseries("Matchs joués sans prédiction préalable", [sql(Q_UNPREDICTED)],
                     "À chaque lot ingéré : ses matchs avaient-ils été prédits ? Une source en panne pendant le "
                     "scoring en laisse passer", unit="short", stack=True, bars=True,
                     overrides=[override("sans prédiction", color=RED), override("prédits avant", color=GREEN)]),
          )  # fmt: skip

    b.row("Quand réentraîner ? Combien coûte une bonne prédiction ?")
    b.add(barchart("Accuracy selon l'ancienneté du modèle", [sql(Q_MODEL_AGE, table=True)],
                   "Jours simulés entre la fin des données d'entraînement et le match, par tranches de 30 : une baisse "
                   "avec l'âge dit qu'il faut réentraîner plus souvent", x_field="âge du modèle", unit="percentunit",
                   overrides=[override("matchs", unit="short", right=True)]))  # fmt: skip
    b.add(timeseries("Coût CPU d'une bonne prédiction", [prom(f"{API_CPU_15M} / ({CORRECT_15M} > 0)", "CPU-s")],
                     "Secondes de CPU de l'API par prédiction jugée juste (fenêtre de 15 min)", unit="s"))  # fmt: skip
    b.add(barchart("Latence selon la taille du batch", [sql(Q_BATCH, table=True)],
                   "Un batch amortit le coût fixe d'un appel : la latence par match baisse",
                   x_field="matchs par appel", unit="ms", overrides=[override("appels", unit="short", right=True)]),
          24, 8)  # fmt: skip
    return b


BOARDS = {"technique": technique, "metier": metier, "croise": croise}


def render() -> dict[str, str]:
    """Le JSON de chaque dashboard, tel qu'il est enregistré dans le dépôt."""
    return {name: json.dumps(build().json(), ensure_ascii=False, indent=2) + "\n" for name, build in BOARDS.items()}


def main(argv: list[str] | None = None) -> None:
    """Écrit (ou vérifie, `--check`) les fichiers JSON des dashboards."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.monitoring.dashboards")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true", help="échoue si un fichier n'est pas à jour")
    args = parser.parse_args(argv)
    stale = []
    for name, content in render().items():
        path = args.out / f"{name}.json"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"{path} : {len(json.loads(content)['panels'])} panneaux")
    if stale:
        raise SystemExit(f"dashboards pas à jour : {', '.join(stale)} (make dashboards)")


if __name__ == "__main__":
    main()
