"""Les dashboards du dépôt : à jour, bien formés, et branchés sur des métriques qui existent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tennis_mlops.api.metrics import ApiMetrics
from tennis_mlops.monitoring import dashboards
from tennis_mlops.monitoring.check import queries

ROOT = Path(__file__).resolve().parents[3]
BOARDS = {name: json.loads(content) for name, content in dashboards.render().items()}
EXPORTER = "".join((ROOT / f"src/tennis_mlops/monitoring/{name}.py").read_text(encoding="utf-8")
                   for name in ("exporter", "containers"))  # fmt: skip


def _api_metrics() -> set[str]:
    names: set[str] = set()
    suffixes = {"counter": ("_total",), "histogram": ("_bucket", "_count", "_sum"), "gauge": ("",)}
    for family in ApiMetrics().registry.collect():
        names.update(family.name + suffix for suffix in suffixes.get(family.type, ("",)))
    return names


def _panels(board: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in board["panels"] if p["type"] != "row"]


def test_committed_json_is_up_to_date() -> None:
    """`make dashboards` a été lancé après la dernière modification de dashboards.py."""
    for name, content in dashboards.render().items():
        path = ROOT / dashboards.OUT / f"{name}.json"
        assert path.read_text(encoding="utf-8") == content, f"{path} pas à jour : make dashboards"


def test_main_writes_and_checks(tmp_path: Path) -> None:
    dashboards.main(["--out", str(tmp_path)])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["croise.json", "metier.json", "technique.json"]
    dashboards.main(["--out", str(tmp_path), "--check"])
    (tmp_path / "metier.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"metier\.json"):
        dashboards.main(["--out", str(tmp_path), "--check"])


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_layout_fits_the_grid_without_overlap(name: str) -> None:
    board = BOARDS[name]
    ids = [p["id"] for p in board["panels"]]
    assert len(ids) == len(set(ids))
    cells: set[tuple[int, int]] = set()
    for panel in board["panels"]:
        pos = panel["gridPos"]
        assert pos["x"] >= 0, panel["title"]
        assert pos["x"] + pos["w"] <= 24, panel["title"]
        mine = {(pos["x"] + dx, pos["y"] + dy) for dx in range(pos["w"]) for dy in range(pos["h"])}
        assert not cells & mine, f"{panel['title']} chevauche un autre panneau"
        cells |= mine


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_every_query_targets_a_provisioned_datasource(name: str) -> None:
    provisioned = yaml.safe_load((ROOT / "infra/grafana/provisioning/datasources/datasources.yml").read_text())
    uids = {d["uid"]: d["type"] for d in provisioned["datasources"]}
    for label, query in queries(BOARDS[name]):
        datasource = query["datasource"]
        assert uids.get(datasource["uid"]) == datasource["type"], label
        if datasource["uid"] == "tennis-pg":
            raw = query["rawSql"]
            assert "$__timeFilter(" in raw or "online_metrics" in raw or "model_data_end" in raw, label
            if query["format"] == "time_series":
                assert re.search(r"\bAS time\b", raw), f"{label} : une série temporelle SQL a une colonne time"


def test_promql_only_uses_metrics_that_exist() -> None:
    """Une faute de frappe dans un nom de métrique donnerait un panneau vide, sans erreur : on la refuse ici."""
    exporter = set(re.findall(r'"(tennis_\w+)"', EXPORTER))
    known = _api_metrics() | exporter | {f"{name}_total" for name in exporter}  # compteurs : suffixe _total
    for board in BOARDS.values():
        for label, query in queries(board):
            if "expr" not in query:
                continue
            used = set(re.findall(r"\b(tennis_\w+?)(?:\{|\[|\s|\)|$)", query["expr"]))
            assert used, label
            assert used <= known, f"{label} : métrique inconnue {used - known}"


def test_business_board_runs_on_simulated_time() -> None:
    metier = BOARDS["metier"]
    assert metier["time"] == dashboards.SIMULATED_RANGE
    sql = [q["rawSql"] for _, q in queries(metier) if "rawSql" in q]
    assert all("match_date" in q or "simulated_date" in q for q in sql)
    # Les dashboards en temps réel annotent à l'heure de l'événement, le métier à la date simulée.
    technique = [a["target"]["rawSql"] for a in BOARDS["technique"]["annotations"]["list"] if "target" in a]
    assert all("e.occurred_at AS time" in q for q in technique)
    simulated = [a["target"]["rawSql"] for a in metier["annotations"]["list"] if "target" in a]
    assert all("COALESCE(e.simulated_date" in q for q in simulated)


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_annotations_cover_every_severity(name: str) -> None:
    groups = [a for a in BOARDS[name]["annotations"]["list"] if "target" in a]
    assert [a["iconColor"] for a in groups] == ["red", "orange", "green", "blue"]
    for severity in ("failure", "warning", "success", "info"):
        assert any(f"e.severity = '{severity}'" in a["target"]["rawSql"] for a in groups)


def test_every_panel_explains_itself_or_is_obvious() -> None:
    """Pas de panneau sans titre ; les chiffres clés ont une description (visible au survol)."""
    for board in BOARDS.values():
        for panel in _panels(board):
            assert panel["title"]
            if panel["type"] in {"bargauge", "barchart", "state-timeline", "table"}:
                assert panel.get("description"), f"{board['title']} · {panel['title']}"


def test_provisioning_points_at_the_dashboards() -> None:
    provider = yaml.safe_load((ROOT / "infra/grafana/provisioning/dashboards/dashboards.yml").read_text())
    path = provider["providers"][0]["options"]["path"]
    compose = yaml.safe_load((ROOT / "compose.yml").read_text())
    mounts = compose["services"]["grafana"]["volumes"]
    assert f"./infra/grafana/dashboards:{path}:ro" in mounts
    home = compose["services"]["grafana"]["environment"]["GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH"]
    assert home == f"{path}/technique.json"
    prometheus = yaml.safe_load((ROOT / "infra/prometheus/prometheus.yml").read_text())
    jobs = {job["job_name"] for job in prometheus["scrape_configs"]}
    assert jobs == {"api", "exporter", "prometheus"}
    exporter = compose["services"]["exporter"]
    assert "/var/run/docker.sock:/var/run/docker.sock:ro" in exporter["volumes"]
    assert "cadvisor" not in compose["services"]
