from datetime import UTC, datetime
from pathlib import Path

import pytest

from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.download import RawDataDownloader
from tennis_mlops.data.errors import DataIntegrityError, SourceUnavailableError
from tennis_mlops.data.lockfile import DataLock, FileRecord, RawManifest

from .fakes import InMemorySource

CATALOG = DatasetCatalog(2020, 2020)
CONTENTS = {name: f"contenu de {name}\n".encode() for name in CATALOG.names()}
FIXED_NOW = datetime(2026, 11, 5, tzinfo=UTC)


def make_downloader(raw_dir: Path, *sources: InMemorySource, lock: DataLock | None = None) -> RawDataDownloader:
    return RawDataDownloader(CATALOG, sources, raw_dir, lock, clock=lambda: FIXED_NOW)


def lock_for(contents: dict[str, bytes], tmp_path: Path, revisions: dict[str, str] | None = None) -> DataLock:
    records = {}
    for name, content in contents.items():
        path = tmp_path / f"ref-{name}"
        path.write_bytes(content)
        records[name] = FileRecord.of(path)
    return DataLock.build(revisions or {}, records)


def test_first_download_writes_files_and_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    source = InMemorySource("hf", CONTENTS)
    report = make_downloader(raw, source).download()

    assert set(report.downloaded) == set(CATALOG.names())
    assert report.reused == ()
    assert (raw / "atp_players.csv").read_bytes() == CONTENTS["atp_players.csv"]
    manifest = RawManifest.load(raw / "manifest.json")
    assert (manifest.source, manifest.revision, manifest.downloaded_at) == ("hf", "a" * 40, FIXED_NOW)
    assert not list(raw.glob("*.part"))


def test_files_matching_the_lock_are_reused(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    lock = lock_for(CONTENTS, tmp_path)
    make_downloader(raw, InMemorySource("hf", CONTENTS), lock=lock).download()

    second = InMemorySource("hf", CONTENTS)
    report = make_downloader(raw, second, lock=lock).download()
    assert second.fetched == []
    assert set(report.reused) == set(CATALOG.names())


def test_pinned_revision_from_lock_is_used(tmp_path: Path) -> None:
    lock = lock_for(CONTENTS, tmp_path, revisions={"hf": "f" * 40})
    source = InMemorySource("hf", CONTENTS, revision="0" * 40)
    make_downloader(tmp_path / "raw", source, lock=lock).download()
    assert {revision for _, revision in source.fetched} == {"f" * 40}


def test_corrupted_content_is_blocking_and_leaves_nothing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    lock = lock_for(CONTENTS, tmp_path)
    corrupted = {**CONTENTS, "atp_players.csv": b"corrompu"}
    fallback = InMemorySource("gh", CONTENTS)

    with pytest.raises(DataIntegrityError, match=r"atp_players\.csv"):
        make_downloader(raw, InMemorySource("hf", corrupted), fallback, lock=lock).download()
    assert fallback.fetched == []  # une corruption ne déclenche jamais de fallback
    assert not (raw / "atp_players.csv").exists()
    assert not list(raw.glob("*.part"))


def test_unavailable_source_falls_back_to_next(tmp_path: Path) -> None:
    primary = InMemorySource("hf", CONTENTS, available=False)
    report = make_downloader(tmp_path / "raw", primary, InMemorySource("gh", CONTENTS)).download()
    assert report.manifest.source == "gh"
    assert report.failed_sources == ("hf",)


def test_all_sources_unavailable(tmp_path: Path) -> None:
    sources = (InMemorySource("hf", CONTENTS, available=False), InMemorySource("gh", CONTENTS, available=False))
    with pytest.raises(SourceUnavailableError, match="hf, gh"):
        make_downloader(tmp_path / "raw", *sources).download()


def test_build_lock_resolves_every_reachable_source(tmp_path: Path) -> None:
    offline = InMemorySource("offline", CONTENTS, available=False)
    downloader = make_downloader(
        tmp_path / "raw", InMemorySource("hf", CONTENTS), InMemorySource("gh", CONTENTS, "b" * 40), offline
    )
    lock = downloader.build_lock(downloader.download().manifest)
    assert lock.revisions == {"gh": "b" * 40, "hf": "a" * 40}
    assert set(lock.files) == set(CATALOG.names())


def test_verify_reports_missing_and_modified_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    lock = lock_for(CONTENTS, tmp_path)
    downloader = make_downloader(raw, InMemorySource("hf", CONTENTS), lock=lock)
    downloader.download()
    assert downloader.verify().dataset_version == lock.dataset_version

    (raw / "atp_players.csv").write_bytes(b"modifie")
    (raw / "atp_matches_2020.csv").unlink()
    with pytest.raises(DataIntegrityError) as excinfo:
        downloader.verify()
    assert "atp_matches_2020.csv manquant" in str(excinfo.value)
    assert "atp_players.csv" in str(excinfo.value)


def test_verify_requires_a_lock(tmp_path: Path) -> None:
    with pytest.raises(DataIntegrityError, match="make data-lock"):
        make_downloader(tmp_path, InMemorySource("hf", CONTENTS)).verify()


def test_at_least_one_source_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source"):
        RawDataDownloader(CATALOG, [], tmp_path)
