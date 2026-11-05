import pytest

from tennis_mlops.data.score import MatchStatus, classify_score


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        ("6-4 7-6(5)", MatchStatus.COMPLETED),
        ("6-7(15) 7-6(2) 6-3 4-6 7-5", MatchStatus.COMPLETED),
        ("6-3 3-6 [10-8]", MatchStatus.COMPLETED),
        ("6-4 2-1 RET", MatchStatus.RETIRED),
        ("RET", MatchStatus.RETIRED),
        ("W/O", MatchStatus.WALKOVER),
        ("Walkover", MatchStatus.WALKOVER),
        ("6-4 3-2 DEF", MatchStatus.DEFAULTED),
        ("2-6 6-3 0-3 Default", MatchStatus.DEFAULTED),
        ("6-3 6-7(5) 2-1 Def.", MatchStatus.DEFAULTED),
        ("6-5 Def.", MatchStatus.DEFAULTED),
        ("7-6(3) 6-4", MatchStatus.COMPLETED),
        ("6-4 5-6 Played and abandoned", MatchStatus.ABANDONED),
        ("2-2 ABD", MatchStatus.ABANDONED),
        ("", MatchStatus.UNKNOWN),
        (None, MatchStatus.UNKNOWN),
        ("n'importe quoi", MatchStatus.UNKNOWN),
        ("6-4 7", MatchStatus.UNKNOWN),
    ],
)
def test_classify_score(score: str | None, expected: MatchStatus) -> None:
    assert classify_score(score) is expected
