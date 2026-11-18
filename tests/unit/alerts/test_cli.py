"""`python -m tennis_mlops.alerts recent|serve` : historique lisible, service assemblé depuis les réglages."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from tennis_mlops.alerts.__main__ import EXIT_FAILED, EXIT_SENT, main
from tennis_mlops.config import Settings
from tennis_mlops.orchestration.events import EventKind, PipelineEvent, PipelineEvents, Severity


def settings(tmp_path: Path) -> Settings:
    return Settings(postgres_dsn=SecretStr(f"sqlite:///{tmp_path / 'tennis.sqlite'}"), teams_webhook_url=None,
                    alert_host="mac.local")  # fmt: skip


def test_recent_alerts_are_listed_with_their_fate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = settings(tmp_path)
    assert main(["recent"], config) == EXIT_SENT and "aucune alerte" in capsys.readouterr().out
    events = PipelineEvents.from_url(config.sqlalchemy_url())
    for title, delivery in [("❌ Échec de l'entraînement", "sent"), ("🚫 Lot refusé", "failed")]:
        events.record(PipelineEvent(EventKind.ALERT, Severity.FAILURE, title, ref=title,
                                    payload={"delivery": delivery, "detail": "webhook : HTTP 400"}))  # fmt: skip
    assert main(["recent", "--limit", "5"], config) == EXIT_SENT
    lines = capsys.readouterr().out.splitlines()
    assert "ÉCHEC" in lines[0] and "🚫 Lot refusé" in lines[0] and "webhook : HTTP 400" in lines[1]
    assert "envoyée" in lines[2] and "Échec de l'entraînement" in lines[2]


def test_unreachable_database(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = Settings(postgres_dsn=SecretStr(f"sqlite:///{tmp_path / 'absent' / 'x.sqlite'}"))
    assert main(["recent"], broken) == EXIT_FAILED and "base injoignable" in capsys.readouterr().err


def test_serve_builds_the_service_from_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: started.update(app=app, **kw))
    assert main(["serve", "--host", "0.0.0.0", "--port", "8091"], settings(tmp_path)) == EXIT_SENT
    assert (started["host"], started["port"]) == ("0.0.0.0", 8091)
    health = TestClient(started["app"]).get("/health").json()
    assert health["teams"] is False and health["silence"]["active"] is False
    assert health["teams_events"] == ["degrade", "echec", "quarantaine", "rollback"]
