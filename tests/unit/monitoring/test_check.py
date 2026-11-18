"""`make grafana-check` contre un Grafana et un Prometheus simulés."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tennis_mlops.monitoring import check, dashboards
from tennis_mlops.monitoring.check import Checker, queries

BOARDS = {b["uid"]: b for b in (json.loads(c) for c in dashboards.render().values())}
PROMETHEUS_JOBS = ["api", "exporter", "prometheus"]


def fake(broken_ref: str | None = None, down_job: str | None = None, stale: bool = False,
         empty_sql: bool = False, docker: bool = True) -> tuple[httpx.MockTransport, list[dict[str, Any]]]:  # fmt: skip
    """Grafana et Prometheus : tout va bien, sauf ce qu'on demande de casser."""
    asked: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "prometheus" and path == "/api/v1/query":
            assert request.url.params["query"] == check.CONTAINERS
            result = [{"metric": {}, "value": [0, "14"]}] if docker else []
            return httpx.Response(200, json={"data": {"result": result}})
        if request.url.host == "prometheus":
            targets = [{"labels": {"job": job}, "health": "down" if job == down_job else "up",
                        "lastError": "connection refused" if job == down_job else "",
                        "scrapeUrl": f"http://{job}/metrics"} for job in PROMETHEUS_JOBS]  # fmt: skip
            return httpx.Response(200, json={"data": {"activeTargets": targets}})
        if path == "/api/health":
            return httpx.Response(200, json={"database": "ok"})
        if path.startswith("/api/datasources/uid/"):
            return httpx.Response(200, json={"status": "OK", "message": "Data source is working"})
        if path.startswith("/api/dashboards/uid/"):
            board = dict(BOARDS[path.rsplit("/", 1)[-1]])
            if stale:
                board["panels"] = board["panels"][:-1]
            return httpx.Response(200, json={"dashboard": board})
        if path == "/api/ds/query":
            body = json.loads(request.content)
            asked.append(body)
            query = body["queries"][0]
            ref = query["refId"]
            if "rawSql" in query and query["rawSql"] == broken_ref:
                return httpx.Response(
                    400, json={"results": {ref: {"error": "db query error: column x does not exist"}}}
                )
            rows = [] if empty_sql and "rawSql" in query else [1.0, 2.0]
            frames = [{"schema": {"fields": [{"name": "time"}, {"name": "v"}]}, "data": {"values": [rows, rows]}}]
            return httpx.Response(200, json={"results": {ref: {"frames": frames}}})
        return httpx.Response(404)

    return httpx.MockTransport(handle), asked


def _run(transport: httpx.MockTransport) -> list[check.Result]:
    return Checker("http://grafana", "http://prometheus", httpx.Client(transport=transport)).run()


def test_everything_green_runs_every_query_of_every_dashboard() -> None:
    transport, asked = fake()
    results = _run(transport)
    assert [r for r in results if r.status != "ok"] == []
    expected = sum(len(list(queries(b))) for b in BOARDS.values())
    assert len(asked) == expected
    # Le dashboard métier est interrogé sur sa période simulée (dates absolues en millisecondes).
    metier = [a for a in asked if a["from"] == "1451606400000"]
    assert metier and all(a["to"] == "1780272000000" for a in metier)
    assert any(a["from"] == "now-1h" for a in asked)


def test_a_broken_query_is_reported_with_grafana_message() -> None:
    broken = next(q for _, q in queries(BOARDS["tennis-metier"]) if "rawSql" in q)["rawSql"]
    results = _run(fake(broken_ref=broken)[0])
    errors = [r for r in results if r.status == "error"]
    assert len(errors) >= 1
    assert "column x does not exist" in errors[0].detail
    assert errors[0].label.startswith("Tennis · Métier · ")


def test_empty_results_warn_without_failing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, _ = fake(empty_sql=True)
    results = _run(transport)
    assert not [r for r in results if r.status == "error"]
    assert any(r.status == "empty" for r in results)
    real_client = httpx.Client
    monkeypatch.setattr(
        "tennis_mlops.monitoring.check.httpx.Client", lambda **kw: real_client(transport=transport, **kw)
    )
    monkeypatch.setenv("TENNIS_GRAFANA_UI_URL", "http://grafana")
    monkeypatch.setenv("TENNIS_PROMETHEUS_URL", "http://prometheus")
    with pytest.raises(SystemExit) as exit_info:
        check.main(["--quiet"])
    assert exit_info.value.code == 0
    assert "sans donnée · 0 erreur(s)" in capsys.readouterr().out


def test_target_down_and_stale_dashboard_fail() -> None:
    results = _run(fake(down_job="exporter", stale=True, docker=False)[0])
    errors = {r.label: r.detail for r in results if r.status == "error"}
    assert errors["Prometheus · cible exporter"] == "connection refused"
    assert "docker.sock" in errors["Exporteur · conteneurs"]
    assert "diffère des fichiers" in errors["Tennis · Technique"]


def test_grafana_down_stops_early() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "grafana":
            raise httpx.ConnectError("refusé", request=request)
        return httpx.Response(200, json={"data": {"activeTargets": []}})

    results = _run(httpx.MockTransport(handle))
    assert [r.label for r in results if r.status == "error"][-1] == "Grafana"
    assert not any(r.label.startswith("Tennis") for r in results)


def test_reads_the_repository_files(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(json.dumps(BOARDS["tennis-croise"]), encoding="utf-8")
    checker = Checker("http://grafana", "http://prometheus", httpx.Client(transport=fake()[0]), tmp_path)
    titles = {r.label.split(" · ")[0] for r in checker.run() if r.label.startswith("Tennis")}
    assert titles == {"Tennis"}
