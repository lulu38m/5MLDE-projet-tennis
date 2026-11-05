import json
from pathlib import Path

import pandas as pd

from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.profile import RawDataProfiler


def write_raw_files(raw: Path) -> DatasetCatalog:
    raw.mkdir()
    catalog = DatasetCatalog(2019, 2020)
    base = {"winner_rank": 1, "loser_rank": 50, "minutes": 90, "best_of": 3, "round": "R32"}
    pd.DataFrame(
        [
            {"tourney_id": "2019-1", "match_num": 1, "tourney_date": 20190107, "tourney_level": "A",
             "surface": "Hard", "score": "6-4 6-4", **base},
            {"tourney_id": "2019-1", "match_num": 2, "tourney_date": 20190107, "tourney_level": "A",
             "surface": "Hard", "score": "W/O", **base, "minutes": None},
            {"tourney_id": "2019-2", "match_num": 1, "tourney_date": 20190520, "tourney_level": "G",
             "surface": "Clay", "score": "6-4 2-1 RET", **base},
        ]
    ).to_csv(raw / "atp_matches_2019.csv", index=False)  # fmt: skip
    pd.DataFrame(
        [
            {"tourney_id": "2020-1", "match_num": 1, "tourney_date": 20200106, "tourney_level": "M",
             "surface": None, "score": "6-3 3-6 [10-8]", **base},
            {"tourney_id": "2020-1", "match_num": 1, "tourney_date": 20200106, "tourney_level": "M",
             "surface": "Grass", "score": None, **base},
        ]
    ).to_csv(raw / "atp_matches_2020.csv", index=False)  # fmt: skip
    pd.DataFrame({"player_id": [1, 2, 3]}).to_csv(raw / "atp_players.csv", index=False)
    rankings = pd.DataFrame({"ranking_date": [20190107, 20190114], "rank": [1, 2], "player": [1, 2], "points": [9, 8]})
    rankings.to_csv(raw / "atp_rankings_10s.csv", index=False)
    rankings.assign(ranking_date=20200106).to_csv(raw / "atp_rankings_20s.csv", index=False)
    rankings.assign(ranking_date=20200113).to_csv(raw / "atp_rankings_current.csv", index=False)
    return catalog


def test_profile_summarises_matches_players_and_rankings(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog = write_raw_files(raw)
    profile = RawDataProfiler(catalog, raw).profile("abc123def456")

    assert profile.total_matches == 5
    assert profile.duplicate_match_keys == 1
    y2019, y2020 = profile.years
    assert (y2019.year, y2019.matches, y2019.tournaments) == (2019, 3, 2)
    assert (y2019.first_tourney_date, y2019.last_tourney_date) == ("20190107", "20190520")
    assert y2019.levels == {"A": 2, "G": 1}
    assert y2020.surfaces == {"unknown": 1, "Grass": 1}
    assert profile.statuses == {"completed": 2, "walkover": 1, "retired": 1, "unknown": 1}
    assert profile.null_rate_pct["minutes"] == 20.0
    assert profile.players == 3
    assert (profile.rankings.rows, profile.rankings.weeks) == (6, 4)
    assert (profile.rankings.first_date, profile.rankings.last_date) == ("20190107", "20200113")

    output = tmp_path / "profile.json"
    profile.save(output)
    assert json.loads(output.read_text())["dataset_version"] == "abc123def456"
