from datetime import date

from mock_source.chaos import break_schema
from tennis_mlops.ingestion.client import MockSourceClient, SourcePayload
from tennis_mlops.ingestion.lots import RESULT_COLUMNS, LotKind, bootstrap_lot, results_frame, tournament_lot
from tests.support.mock_world import MockEnv


def lot_payload(env: MockEnv) -> SourcePayload:
    env.at(date(2019, 1, 14))
    return MockSourceClient("", 5, http=env.client).tournament_results("2019-0001")


def test_tournament_lot_is_flattened_with_rank_gap(env: MockEnv) -> None:
    lot = tournament_lot(lot_payload(env))
    assert (lot.lot_id, lot.kind, lot.tourney_level, lot.expected_rows, lot.rows) == (
        "2019-0001",
        LotKind.TOURNAMENT,
        "A",
        7,
        7,
    )
    assert lot.end_date == date(2019, 1, 13)
    assert lot.simulated_now == date(2019, 1, 14)
    assert set(RESULT_COLUMNS) <= set(lot.frame.columns)
    final = lot.frame.set_index("match_id").loc["2019-0001-7"]
    assert (final["winner_id"], final["loser_id"], final["rank_gap"]) == (1, 2, 1)


def test_bootstrap_lot_id_names_its_period(env: MockEnv) -> None:
    payload = lot_payload(env)
    lot = bootstrap_lot(payload, date(2019, 1, 1), date(2019, 2, 1))
    assert (lot.lot_id, lot.kind, lot.expected_rows) == ("bootstrap_2019-01-01_2019-02-01", LotKind.BOOTSTRAP, None)


def test_empty_and_broken_payloads_are_kept_as_received(env: MockEnv) -> None:
    assert list(results_frame([]).columns) == list(RESULT_COLUMNS)
    broken = results_frame(break_schema(lot_payload(env).data)["results"])
    assert "matchId" in broken.columns and "match_id" not in broken.columns
    assert "score" not in broken.columns
    assert "winner_player_id" in broken.columns
