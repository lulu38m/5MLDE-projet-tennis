"""Stockage : SQLite toujours ; PostgreSQL en plus si `TENNIS_TEST_POSTGRES_URL` est défini (marqueur integration)."""

import json
import os
from collections.abc import Iterator
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
import sqlalchemy as sa

from tennis_mlops.ingestion.archive import LotArchive
from tennis_mlops.ingestion.client import MockSourceClient
from tennis_mlops.ingestion.lots import Lot, tournament_lot
from tennis_mlops.ingestion.store import IngestionStore, LotStatus, metadata
from tennis_mlops.validation.validator import CheckResult, ValidationReport
from tests.support.mock_world import MockEnv

POSTGRES_URL = os.environ.get("TENNIS_TEST_POSTGRES_URL")
BACKENDS = ["sqlite", pytest.param("postgres", marks=pytest.mark.integration)] if POSTGRES_URL else ["sqlite"]


@pytest.fixture(params=BACKENDS)
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[IngestionStore]:
    url = POSTGRES_URL if request.param == "postgres" else f"sqlite:///{tmp_path / 'test.sqlite'}"
    assert url is not None
    engine = sa.create_engine(url)
    metadata.drop_all(engine)
    store = IngestionStore(engine)
    store.create_schema()
    yield store
    metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def lot(env: MockEnv) -> Lot:
    env.at(date(2019, 1, 14))
    return tournament_lot(MockSourceClient("", 5, http=env.client).tournament_results("2019-0001"))


OK = ValidationReport(lot_id="2019-0001", success=True, checks=[], docs_index=None)
KO = ValidationReport(
    lot_id="2019-0001",
    success=False,
    checks=[CheckResult("expect_tennis_score_to_be_valid", "score", False, True, 1, ["6-4 7-"])],
    docs_index="state/gx/index.html",
)


def test_watermark_defaults_then_moves(store: IngestionStore) -> None:
    assert store.watermark(default=date(2019, 1, 1)) == date(2019, 1, 1)
    store.set_watermark(date(2019, 2, 1))
    store.set_watermark(date(2019, 3, 1))
    assert store.watermark(default=date(2019, 1, 1)) == date(2019, 3, 1)


def test_valid_lot_is_stored_once(store: IngestionStore, lot: Lot) -> None:
    assert store.save_valid_lot(lot, OK, "state/lots/2019-0001", "run-1") == 7
    assert store.save_valid_lot(lot, OK, "state/lots/2019-0001", "run-2") == 0  # reprise sans doublon
    assert store.count_matches() == 7
    assert store.valid_lot_ids() == {"2019-0001"}
    trace = store.lot("2019-0001")
    assert trace is not None
    assert (trace["status"], trace["rows_received"], trace["rows_expected"], trace["flow_run_id"]) == (
        LotStatus.VALID,
        7,
        7,
        "run-2",
    )


def test_quarantined_lot_stores_nothing_then_can_be_replayed(store: IngestionStore, lot: Lot) -> None:
    store.save_quarantined_lot(lot, KO, "state/quarantine/2019-0001", "run-1")
    trace = store.lot("2019-0001")
    assert trace is not None and trace["status"] == LotStatus.QUARANTINED
    assert json.loads(trace["failed_checks"]) == ["expect_tennis_score_to_be_valid(score) 1 ligne(s), ex. ['6-4 7-']"]
    assert store.count_matches() == 0 and store.valid_lot_ids() == set()
    store.save_valid_lot(lot, OK, None, "run-2")
    trace = store.lot("2019-0001")
    assert trace is not None and trace["status"] == LotStatus.VALID
    assert store.lot("inconnu") is None


def test_stored_values_keep_their_types(store: IngestionStore, lot: Lot) -> None:
    store.save_valid_lot(lot, OK, None, None)
    with store._engine.connect() as conn:
        row = conn.execute(sa.text("SELECT * FROM matches WHERE match_id = '2019-0001-3'")).mappings().one()
    assert (row["status"], row["score"], row["minutes"], row["loser_entry"], row["loser_ht"]) == (
        "walkover",
        "W/O",
        90,
        "Q",
        None,
    )
    assert str(row["match_date"]) == "2019-01-07"


def test_rank_gap_reference_needs_enough_history(store: IngestionStore, lot: Lot) -> None:
    assert store.rank_gap_reference("A", date(2020, 1, 1), 1, (0.0, 1.0)) is None
    store.save_valid_lot(lot, OK, None, None)
    other = replace(
        lot, lot_id="autre", frame=lot.frame.assign(tourney_id="autre", match_id=lot.frame["match_id"] + "b")
    )
    store.save_valid_lot(other, OK, None, None)
    reference = store.rank_gap_reference("A", date(2020, 1, 1), 2, (0.0, 1.0))
    assert reference is not None and (reference.level, reference.tournaments) == ("A", 2)
    assert reference.low <= reference.high
    assert store.rank_gap_reference("A", date(2020, 1, 1), 3, (0.0, 1.0)) is None
    assert store.rank_gap_reference("A", date(2019, 1, 1), 1, (0.0, 1.0)) is None  # rien avant cette date


def test_archive_separates_accepted_and_quarantined_lots(tmp_path: Path, lot: Lot) -> None:
    archive = LotArchive(tmp_path)
    accepted = archive.save(lot, OK)
    refused = archive.save(lot, KO)
    assert accepted == tmp_path / "lots" / "2019-0001"
    assert refused == tmp_path / "quarantine" / "2019-0001"
    envelope = json.loads((refused / "payload.json").read_text())
    assert envelope["endpoint"] == "/tournaments/2019-0001/results"
    assert envelope["data"]["expected_matches"] == 7
    assert json.loads((refused / "validation.json").read_text())["success"] is False
