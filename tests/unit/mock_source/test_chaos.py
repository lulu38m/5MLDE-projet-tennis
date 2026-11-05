from datetime import date
from typing import Any

from mock_source.chaos import ChaosController, ChaosMode, ChaosSettings, break_schema, corrupt_payload


def results_payload(n: int) -> dict[str, Any]:
    return {
        "count": n,
        "results": [
            {
                "match_id": f"2019-0001-{i}",
                "match_date": "2019-01-07",
                "surface": "Hard",
                "winner": {"id": 1, "name": "A"},
                "loser": {"id": 2, "name": "B"},
                "score": "6-4 6-4",
                "minutes": 90,
            }
            for i in range(n)
        ],
    }


def test_failures_counter_returns_to_off() -> None:
    chaos = ChaosController()
    chaos.configure(ChaosSettings(mode=ChaosMode.UNAVAILABLE, failures=2))
    assert [chaos.consume().mode for _ in range(3)] == [ChaosMode.UNAVAILABLE, ChaosMode.UNAVAILABLE, ChaosMode.OFF]
    assert chaos.state().mode is ChaosMode.OFF


def test_unlimited_mode_stays_active() -> None:
    chaos = ChaosController()
    chaos.configure(ChaosSettings(mode=ChaosMode.SLOW, delay_s=0.1))
    assert all(chaos.consume().mode is ChaosMode.SLOW for _ in range(5))


def test_corruption_is_deterministic_and_covers_every_defect() -> None:
    payload = results_payload(200)
    first = corrupt_payload(payload, date(2019, 1, 20))
    assert first == corrupt_payload(payload, date(2019, 1, 20))
    records = first["results"]
    assert first["count"] == len(records) < 200  # lignes perdues
    assert any(r["winner"]["id"] == r["loser"]["id"] for r in records)
    assert any(r["score"] == "6-4 7-" for r in records)
    assert any(r["minutes"] < 0 for r in records)
    assert any(r["match_date"] > "2019-01-20" for r in records)
    assert any(r["surface"] == "Moon" for r in records)
    assert payload["results"][0]["winner"]["id"] == 1  # l'original n'est pas modifié
    assert all(r["loser"]["id"] == 2 for r in payload["results"])


def test_schema_break_renames_drops_and_retypes() -> None:
    broken = break_schema(results_payload(1))["results"][0]
    assert "matchId" in broken and "match_id" not in broken
    assert "score" not in broken
    assert broken["minutes"] == "90 min"
    assert "player_id" in broken["winner"] and "id" not in broken["winner"]


def test_payload_without_matches_is_untouched() -> None:
    payload = {"count": 0, "tournaments": []}
    assert corrupt_payload(payload, date(2019, 1, 1)) == payload
    assert break_schema(payload) == payload
