import pytest

from tennis_mlops.config import ProjectConfig
from tennis_mlops.data.catalog import DatasetCatalog, FileKind


def test_files_cover_years_decades_players_and_docs() -> None:
    catalog = DatasetCatalog(2010, 2026)
    names = catalog.names()
    assert names[0] == "atp_matches_2010.csv"
    assert len(catalog.of_kind(FileKind.MATCHES)) == 17
    assert [f.name for f in catalog.of_kind(FileKind.RANKINGS)] == [
        "atp_rankings_10s.csv",
        "atp_rankings_20s.csv",
        "atp_rankings_current.csv",
    ]
    assert "atp_players.csv" in names
    assert "matches_data_dictionary.txt" in names
    assert len(names) == len(set(names))


def test_decade_name_wraps_century() -> None:
    names = DatasetCatalog(1999, 2001).names()
    assert "atp_rankings_90s.csv" in names
    assert "atp_rankings_00s.csv" in names


def test_match_files_are_keyed_by_year() -> None:
    assert DatasetCatalog(2020, 2021).match_files() == {2020: "atp_matches_2020.csv", 2021: "atp_matches_2021.csv"}


@pytest.mark.parametrize(("first", "last"), [(2020, 2010), (1900, 2010)])
def test_invalid_year_range_is_rejected(first: int, last: int) -> None:
    with pytest.raises(ValueError, match="plage"):
        DatasetCatalog(first, last)


def test_from_config_uses_data_section() -> None:
    catalog = DatasetCatalog.from_config(ProjectConfig.from_yaml().data)
    assert catalog.years == range(2010, 2027)
