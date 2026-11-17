"""`make grafana-check` : les dashboards fonctionnent-ils sur la stack qui tourne ?

Sans ouvrir de navigateur, on vérifie dans l'ordre :

1. Prometheus collecte ses trois cibles (api, exporter, prometheus), et l'exporteur lit bien l'API Docker ;
2. Grafana répond, et ses deux sources de données passent leur test de santé ;
3. les trois dashboards du dépôt sont chargés, à jour (même version que les fichiers) ;
4. chaque requête de chaque panneau et de chaque annotation est exécutée par Grafana lui-même (`/api/ds/query`),
   sur la période du dashboard : une erreur (SQL, PromQL, droits) fait échouer la vérification ; une requête qui
   répond sans donnée est signalée, sans échec (le dashboard Métier est vide tant que rien n'a été prédit).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from tennis_mlops.config import Settings
from tennis_mlops.monitoring.dashboards import OUT

JOBS = ("api", "exporter", "prometheus")
DATASOURCES = ("prometheus", "tennis-pg")
CONTAINERS = "count(tennis_container_memory_bytes) and on() tennis_docker_up == 1"


@dataclass(frozen=True)
class Result:
    """Une ligne du rapport : ✅ ok, ⚠️ vide (sans échec), ❌ erreur."""

    status: str  # "ok" | "empty" | "error"
    label: str
    detail: str = ""

    def line(self) -> str:
        """Affichage."""
        icon = {"ok": "✅", "empty": "⚠️ ", "error": "❌"}[self.status]
        return f"{icon} {self.label}" + (f" : {self.detail}" if self.detail else "")


def queries(board: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Chaque requête d'un dashboard : celles des panneaux, puis celles des annotations."""
    for panel in board.get("panels", []):
        for target in panel.get("targets", []):
            yield f"{panel['title']} [{target.get('refId', 'A')}]", target | {"datasource": panel["datasource"]}
    for annotation in board.get("annotations", {}).get("list", []):
        if "target" in annotation:
            yield (
                f"annotation « {annotation['name']} »",
                annotation["target"] | {"datasource": annotation["datasource"]},
            )


def _bound(value: str) -> str:
    """Bornes de temps pour /api/ds/query : relatives (`now-1h`) telles quelles, dates ISO en millisecondes."""
    if value.startswith("now"):
        return value
    return str(int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000))


def _rows(result: dict[str, Any]) -> int:
    """Lignes renvoyées, toutes séries confondues (une trame = des colonnes de même longueur)."""
    total = 0
    for frame in result.get("frames") or []:
        columns = (frame.get("data") or {}).get("values") or []
        total += len(columns[0]) if columns else 0
    return total


@dataclass
class Checker:
    """Les vérifications, contre Grafana et Prometheus publiés sur la machine hôte."""

    grafana: str
    prometheus: str
    http: httpx.Client
    dashboards: Path = OUT

    def run(self) -> list[Result]:
        """Toutes les vérifications, dans l'ordre."""
        out = list(self.targets())
        out += self.containers()
        out += self.health()
        if any(r.status == "error" and r.label.startswith("Grafana") for r in out):
            return out
        for path in sorted(self.dashboards.glob("*.json")):
            out += self.dashboard(json.loads(path.read_text(encoding="utf-8")))
        return out

    def targets(self) -> Iterator[Result]:
        """Les cibles de Prometheus."""
        try:
            response = self.http.get(f"{self.prometheus}/api/v1/targets", params={"state": "active"})
            response.raise_for_status()
            active = response.json()["data"]["activeTargets"]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            yield Result("error", "Prometheus", f"injoignable sur {self.prometheus} ({exc})")
            return
        health = {t["labels"].get("job"): t for t in active}
        for job in JOBS:
            target = health.get(job)
            if target is None:
                yield Result("error", f"Prometheus · cible {job}", "absente de la configuration")
            elif target.get("health") != "up":
                yield Result("error", f"Prometheus · cible {job}", target.get("lastError") or target.get("health", ""))
            else:
                yield Result("ok", f"Prometheus · cible {job}", target.get("scrapeUrl", ""))

    def containers(self) -> Iterator[Result]:
        """L'exporteur lit-il l'API Docker ? (sinon, ni CPU ni mémoire par conteneur sur les dashboards)."""
        try:
            response = self.http.get(f"{self.prometheus}/api/v1/query", params={"query": CONTAINERS})
            response.raise_for_status()
            result = response.json()["data"]["result"]
        except (httpx.HTTPError, ValueError, KeyError):
            return  # Prometheus injoignable : déjà signalé
        if result:
            yield Result("ok", "Exporteur · conteneurs", f"{int(float(result[0]['value'][1]))} suivis (API Docker)")
        else:
            hint = "API Docker illisible : socket /var/run/docker.sock monté ? (docker compose logs exporter)"
            yield Result("error", "Exporteur · conteneurs", hint)

    def health(self) -> Iterator[Result]:
        """Grafana et ses sources de données."""
        try:
            self.http.get(f"{self.grafana}/api/health").raise_for_status()
        except httpx.HTTPError as exc:
            yield Result("error", "Grafana", f"injoignable sur {self.grafana} ({exc})")
            return
        yield Result("ok", "Grafana", self.grafana)
        for uid in DATASOURCES:
            try:
                response = self.http.get(f"{self.grafana}/api/datasources/uid/{uid}/health")
                body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                yield Result("error", f"Grafana · source {uid}", str(exc))
                continue
            status = "ok" if response.status_code == 200 and body.get("status") == "OK" else "error"
            yield Result(status, f"Grafana · source {uid}", str(body.get("message", "")))

    def dashboard(self, board: dict[str, Any]) -> list[Result]:
        """Un dashboard : chargé et à jour, puis chacune de ses requêtes."""
        title = board["title"]
        response = self.http.get(f"{self.grafana}/api/dashboards/uid/{board['uid']}")
        if response.status_code != 200:
            return [Result("error", title, f"pas chargé par Grafana (HTTP {response.status_code})")]
        loaded = response.json()["dashboard"]
        panels = len(board["panels"])
        if len(loaded.get("panels", [])) != panels:
            return [Result("error", title, "la version chargée diffère des fichiers (Grafana relit toutes les 30 s)")]
        out = [Result("ok", title, f"{panels} panneaux")]
        window = {"from": _bound(board["time"]["from"]), "to": _bound(board["time"]["to"])}
        for label, query in queries(board):
            out.append(self.query(f"{title} · {label}", query, window))
        return out

    def query(self, label: str, query: dict[str, Any], window: dict[str, str]) -> Result:
        """Une requête exécutée par Grafana, comme le ferait le panneau."""
        body = {"queries": [query | {"intervalMs": 60_000, "maxDataPoints": 500}], **window}
        try:
            response = self.http.post(f"{self.grafana}/api/ds/query", json=body)
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return Result("error", label, str(exc))
        result = (data.get("results") or {}).get(query.get("refId", "A"), {})
        error = result.get("error") or (data.get("message") if response.status_code >= 400 else None)
        if error:
            return Result("error", label, str(error).splitlines()[0][:200])
        rows = _rows(result)
        return Result("ok", label, f"{rows} point(s)") if rows else Result("empty", label, "aucune donnée")


def main(argv: list[str] | None = None) -> None:
    """Affiche le rapport ; code de sortie 1 si une vérification échoue."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.monitoring.check", description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="n'afficher que les erreurs et le résumé")
    args = parser.parse_args(argv)
    settings = Settings()
    auth = ("admin", settings.grafana_admin_password.get_secret_value())
    with httpx.Client(timeout=30.0, auth=auth) as http:
        checker = Checker(str(settings.grafana_ui_url).rstrip("/"), str(settings.prometheus_url).rstrip("/"), http)
        results = checker.run()
    for result in results:
        if not args.quiet or result.status == "error":
            print(result.line())
    errors = sum(r.status == "error" for r in results)
    empty = sum(r.status == "empty" for r in results)
    print(f"\n{len(results) - errors - empty} ok · {empty} sans donnée · {errors} erreur(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
