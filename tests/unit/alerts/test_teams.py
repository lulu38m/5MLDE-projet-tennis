from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from tennis_mlops.alerts.__main__ import EXIT_FAILED, EXIT_NOT_CONFIGURED, EXIT_SENT, main
from tennis_mlops.alerts.examples import EXAMPLES
from tennis_mlops.alerts.teams import Delivery, TeamsNotifier
from tennis_mlops.config import Settings

SECRET = "https://example.invalid/workflows/abc?sig=SECRET-SIGNATURE"


def client(status: int, seen: list[httpx.Request] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, text="refusé" if status >= 300 else "")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sent_disabled_and_failed() -> None:
    seen: list[httpx.Request] = []
    alert = EXAMPLES["degrade"]()
    assert TeamsNotifier(SecretStr(SECRET), client=client(202, seen)).send(alert).status is Delivery.SENT
    assert seen[0].method == "POST" and b'"application/vnd.microsoft.card.adaptive"' in seen[0].content
    assert TeamsNotifier(None).send(alert).status is Delivery.DISABLED
    assert not TeamsNotifier(SecretStr("  ")).enabled  # `TENNIS_TEAMS_WEBHOOK_URL=` vide dans .env
    refused = TeamsNotifier(SecretStr(SECRET), client=client(400)).send(alert)
    assert refused.status is Delivery.FAILED and "HTTP 400" in refused.detail


def test_the_webhook_url_never_leaks_in_errors() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect to {request.url}")

    result = TeamsNotifier(SecretStr(SECRET), client=httpx.Client(transport=httpx.MockTransport(down))).send(
        EXAMPLES["rollback"]()
    )
    assert result.status is Delivery.FAILED
    assert "SECRET" not in result.detail and "example.invalid" not in result.detail


def test_command_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["preview", "--out", str(tmp_path)]) == EXIT_SENT
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{n}.json" for n in EXAMPLES)

    assert main(["test"], settings=Settings(teams_webhook_url=None)) == EXIT_NOT_CONFIGURED
    posted: list[dict[str, object]] = []

    def fake_post(url: str, json: dict[str, object], timeout: float) -> httpx.Response:
        posted.append(json)
        return httpx.Response(202)

    monkeypatch.setattr(httpx, "post", fake_post)
    configured = Settings(teams_webhook_url=SecretStr(SECRET))
    assert main(["test", "--example", "echec"], settings=configured) == EXIT_SENT
    header = posted[0]["attachments"][0]["content"]["body"][0]["items"]  # type: ignore[index]
    assert header[0]["text"] == "🧪 TEST · Échec de l'entraînement"
    assert "aucun incident réel" in header[2]["text"]
    assert "SECRET" not in capsys.readouterr().out

    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(500, text="panne"))
    assert main(["test"], settings=configured) == EXIT_FAILED
