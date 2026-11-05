import pytest

from mock_source.schedule import estimate_day_offsets, tournament_duration_days

GRAND_SLAM = ["R128"] * 64 + ["R64"] * 32 + ["R32"] * 16 + ["R16"] * 8 + ["QF"] * 4 + ["SF"] * 2 + ["F"]


def by_round(rounds: list[str], level: str, draw: int) -> dict[str, set[int]]:
    days = estimate_day_offsets(rounds, list(range(1, len(rounds) + 1)), level, draw)
    result: dict[str, set[int]] = {}
    for rnd, day in zip(rounds, days, strict=True):
        result.setdefault(rnd, set()).add(day)
    return result


def test_grand_slam_spreads_rounds_over_two_weeks() -> None:
    days = by_round(GRAND_SLAM, "G", 128)
    ordered = [days[r] for r in ("R128", "R64", "R32", "R16", "QF", "SF", "F")]
    assert all(len(d) == 1 for d in ordered)  # un tour = un jour
    flat = [next(iter(d)) for d in ordered]
    assert flat == sorted(flat) and len(set(flat)) == 7  # strictement croissant
    assert (flat[0], flat[-1]) == (0, 13)


def test_atp_250_three_rounds() -> None:
    days = by_round(["QF"] * 4 + ["SF"] * 2 + ["F"], "A", 8)
    assert days == {"QF": {0}, "SF": {3}, "F": {6}}


def test_round_robin_comes_before_knockout() -> None:
    days = by_round(["RR"] * 12 + ["SF"] * 2 + ["F"], "F", 8)
    assert max(days["RR"]) < min(days["SF"]) < min(days["F"])
    assert days["F"] == {6}


def test_davis_cup_rubbers_over_three_days() -> None:
    assert estimate_day_offsets(["RR"] * 4, [1, 2, 4, 5], "D", 4) == [0, 0, 1, 2]


def test_round_robin_order_follows_match_number_not_list_order() -> None:
    assert estimate_day_offsets(["RR"] * 4, [5, 4, 2, 1], "D", 4) == [2, 1, 0, 0]


def test_bronze_medal_match_is_played_with_the_final() -> None:
    days = by_round(["SF", "SF", "BR", "F"], "O", 64)
    assert days["BR"] == days["F"]


def test_more_rounds_than_days_never_collide() -> None:
    days = by_round(["R32"] * 16 + ["R16"] * 8 + ["QF"] * 4 + ["SF"] * 2 + ["F"], "D", 32)
    flat = [next(iter(days[r])) for r in ("R32", "R16", "QF", "SF", "F")]
    assert len(set(flat)) == 5 and flat == sorted(flat)


@pytest.mark.parametrize(
    ("level", "draw", "rr", "expected"),
    [("G", 128, False, 14), ("M", 96, False, 12), ("M", 56, False, 7), ("O", 64, False, 8), ("D", 4, True, 3),
     ("A", 16, True, 10), ("A", 32, False, 7), ("F", 8, True, 7)],
)  # fmt: skip
def test_durations(level: str, draw: int, rr: bool, expected: int) -> None:
    assert tournament_duration_days(level, draw, rr) == expected


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="longueur"):
        estimate_day_offsets(["F"], [1, 2], "A", 32)
    with pytest.raises(ValueError, match="inconnu"):
        estimate_day_offsets(["Q1"], [1], "A", 32)
