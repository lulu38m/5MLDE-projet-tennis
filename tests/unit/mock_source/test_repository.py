from datetime import date
from pathlib import Path

import pytest

from mock_source.repository import ReplayRepository, player1_is_winner
from mock_source.schemas import TournamentStatus
from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.errors import DataIntegrityError
from tennis_mlops.data.lockfile import DataLock
from tests.support.mock_world import CONFIG, write_dataset


def dates_by_round(repository: ReplayRepository, tourney_id: str) -> dict[str, set[date]]:
    found = repository.tournament(tourney_id, date(2020, 12, 31))
    assert found is not None
    days: dict[str, set[date]] = {}
    for result in found[1]:
        days.setdefault(result.round, set()).add(result.match_date)
    return days


def test_match_dates_are_estimated_from_round(repository: ReplayRepository) -> None:
    assert dates_by_round(repository, "2019-0001") == {
        "QF": {date(2019, 1, 7)},
        "SF": {date(2019, 1, 10)},
        "F": {date(2019, 1, 13)},
    }
    davis = sorted(r.match_date.day for r in repository.results(date(2019, 3, 1), date(2019, 3, 4)))
    assert davis == [1, 1, 2, 3]


def test_results_respect_half_open_interval(repository: ReplayRepository) -> None:
    assert {r.round for r in repository.results(date(2019, 1, 7), date(2019, 1, 10))} == {"QF"}
    assert len(repository.results(date(2019, 1, 7), date(2019, 1, 14))) == 7
    assert repository.results(date(2019, 1, 14), date(2019, 2, 1)) == []


def test_results_carry_status_and_pre_match_info(repository: ReplayRepository) -> None:
    by_num = {r.match_num: r for r in repository.results(date(2019, 1, 1), date(2019, 1, 14))}
    assert by_num[2].status.value == "retired"
    assert by_num[3].status.value == "walkover"
    assert by_num[1].match_id == "2019-0001-1"
    assert (by_num[1].winner.rank, by_num[1].loser.entry, by_num[1].loser.ht) == (1, "Q", None)
    assert by_num[1].winner.age == 25.0


@pytest.mark.parametrize(
    ("now", "expected_rounds"),
    [
        (date(2019, 1, 4), set()),  # tableau pas encore publié (J-2)
        (date(2019, 1, 5), {"QF"}),
        (date(2019, 1, 7), {"QF"}),  # les QF du jour sont encore à venir
        (date(2019, 1, 8), {"SF"}),  # les QF sont finis : les demi-finalistes sont connus
        (date(2019, 1, 11), {"F"}),
        (date(2019, 1, 14), set()),
    ],
)
def test_fixtures_only_reveal_known_players(repository: ReplayRepository, now: date, expected_rounds: set[str]) -> None:
    fixtures = [f for f in repository.fixtures(now, 7) if f.tourney_id == "2019-0001"]
    assert {f.round for f in fixtures} == expected_rounds


def test_fixture_player_order_is_neutral_and_deterministic(repository: ReplayRepository) -> None:
    for fixture in repository.fixtures(date(2019, 1, 5), 7):
        result = next(
            r for r in repository.results(date(2019, 1, 7), date(2019, 1, 8)) if r.match_id == fixture.match_id
        )
        expected_p1 = result.winner if player1_is_winner(fixture.match_id) else result.loser
        assert fixture.player1 == expected_p1


def test_fixture_horizon(repository: ReplayRepository) -> None:
    assert repository.fixtures(date(2019, 1, 5), 1) == []
    assert len(repository.fixtures(date(2019, 1, 5), 3)) == 4


def test_tournament_visibility_and_status(repository: ReplayRepository) -> None:
    assert repository.tournament("2019-0001", date(2018, 12, 23)) is None  # annoncé 14 jours avant
    statuses = {
        day: repository.tournament("2019-0001", day)[0].status  # type: ignore[index]
        for day in (date(2018, 12, 24), date(2019, 1, 8), date(2019, 1, 14))
    }
    assert list(statuses.values()) == [
        TournamentStatus.UPCOMING,
        TournamentStatus.IN_PROGRESS,
        TournamentStatus.COMPLETED,
    ]


def test_tournament_lot_only_contains_played_matches(repository: ReplayRepository) -> None:
    found = repository.tournament("2019-0001", date(2019, 1, 11))
    assert found is not None
    summary, results = found
    assert (summary.expected_matches, summary.completed_matches, len(results)) == (7, 6, 6)
    assert repository.tournament("inconnu", date(2020, 1, 1)) is None


def test_tournament_listing_filters(repository: ReplayRepository) -> None:
    now = date(2020, 1, 20)
    assert [t.tourney_id for t in repository.tournaments(now)] == ["2019-0001", "2019-D001", "2020-0002"]
    assert [t.tourney_id for t in repository.tournaments(now, since=date(2019, 2, 1))] == ["2019-D001", "2020-0002"]
    assert [t.tourney_id for t in repository.tournaments(now, until=date(2019, 2, 1))] == ["2019-0001"]
    completed = repository.tournaments(date(2019, 3, 2), status=TournamentStatus.COMPLETED)
    assert [t.tourney_id for t in completed] == ["2019-0001"]
    assert repository.tournaments_completed_between(date(2019, 1, 1), date(2019, 1, 14)) == ["2019-0001"]


def test_empty_period_reason(repository: ReplayRepository) -> None:
    assert repository.empty_period_reason(date(2019, 5, 1), date(2019, 6, 1)) == "Pause de test"
    assert "intersaison" in repository.empty_period_reason(date(2019, 2, 1), date(2019, 2, 10))


def test_rankings_use_last_published_week(repository: ReplayRepository) -> None:
    assert repository.rankings(date(2019, 1, 6), 10) == (None, [])
    ranking_date, rows = repository.rankings(date(2019, 1, 20), 3)
    assert ranking_date == date(2019, 1, 14)
    assert [(r.rank, r.player_id, r.name) for r in rows] == [(1, 1, "Joueur 1"), (2, 2, "Joueur 2"), (3, 3, "Joueur 3")]
    assert [r.player_id for r in repository.rankings(date(2020, 1, 20), 10, player_id=4)[1]] == [4]


def test_player_history_stops_before_as_of(repository: ReplayRepository) -> None:
    history = repository.player_history(1, date(2019, 1, 11), match_limit=50, ranking_weeks=52)
    assert history is not None
    assert (history.wins, history.losses, history.is_debutant) == (2, 0, False)
    assert [m.round for m in history.matches] == ["SF", "QF"]  # du plus récent au plus ancien
    assert history.matches[0].opponent_name == "Joueur 4"
    assert [p.ranking_date for p in history.rankings] == [date(2019, 1, 7)]
    later = repository.player_history(1, date(2020, 1, 20), match_limit=1, ranking_weeks=520)
    assert later is not None and len(later.matches) == 1 and later.losses == 1


def test_debutant_and_unknown_players(repository: ReplayRepository) -> None:
    debutant = repository.player_history(9, date(2020, 1, 1), 50, 52)
    assert debutant is not None and debutant.is_debutant and debutant.player.dob == date(1994, 1, 1)
    assert repository.player_history(10, date(2020, 1, 1), 50, 52) is None


def test_loading_refuses_altered_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog, lock = write_dataset(raw)
    (raw / "atp_players.csv").write_text("altéré")
    with pytest.raises(DataIntegrityError, match=r"atp_players\.csv"):
        ReplayRepository.from_raw(raw, catalog, lock, CONFIG)


def test_repository_basic_properties(
    repository: ReplayRepository, dataset: tuple[Path, DatasetCatalog, DataLock]
) -> None:
    assert repository.match_count == 14
    assert repository.dataset_version == dataset[2].dataset_version
    assert (repository.first_date, repository.last_date) == (date(2019, 1, 7), date(2020, 1, 12))
    assert repository.count_completed_between(date(2019, 1, 1), date(2019, 1, 8)) == 4
