import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from tennis_mlops.data.errors import DataIntegrityError
from tennis_mlops.data.lockfile import DataLock, FileRecord, RawManifest, compute_dataset_version, sha256_file


def record(content: bytes) -> FileRecord:
    return FileRecord(sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    path.write_bytes(b"x" * 3_000_000)  # plusieurs blocs
    assert sha256_file(path) == hashlib.sha256(b"x" * 3_000_000).hexdigest()
    assert FileRecord.of(path).size_bytes == 3_000_000


def test_dataset_version_ignores_order_and_source() -> None:
    files = {"b.csv": record(b"b"), "a.csv": record(b"a")}
    reordered = {"a.csv": record(b"a"), "b.csv": record(b"b")}
    assert compute_dataset_version(files) == compute_dataset_version(reordered)
    lock_hf = DataLock.build({"huggingface:x/y": "1" * 40}, files)
    lock_gh = DataLock.build({"github:x/y": "2" * 40}, files)
    assert lock_hf.dataset_version == lock_gh.dataset_version
    assert len(lock_hf.dataset_version) == 12


def test_dataset_version_changes_with_content() -> None:
    assert compute_dataset_version({"a.csv": record(b"a")}) != compute_dataset_version({"a.csv": record(b"A")})


def test_lock_roundtrip_is_sorted(tmp_path: Path) -> None:
    lock = DataLock.build({"z:repo/b": "2" * 40, "a:repo/a": "1" * 40}, {"b.csv": record(b"b"), "a.csv": record(b"a")})
    path = tmp_path / "configs" / "data.lock.json"
    lock.save(path)
    raw = json.loads(path.read_text())
    assert list(raw["files"]) == ["a.csv", "b.csv"]
    assert list(raw["revisions"]) == ["a:repo/a", "z:repo/b"]
    assert DataLock.load(path) == lock


def test_tampered_lock_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    DataLock.build({}, {"a.csv": record(b"a")}).save(path)
    raw = json.loads(path.read_text())
    raw["files"]["a.csv"]["sha256"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(ValidationError, match="dataset_version incohérent"):
        DataLock.load(path)


def test_verify_detects_mismatch_and_unknown_files() -> None:
    lock = DataLock.build({}, {"a.csv": record(b"a")})
    lock.verify("a.csv", record(b"a"))
    with pytest.raises(DataIntegrityError, match="SHA-256"):
        lock.verify("a.csv", record(b"corrompu"))
    with pytest.raises(DataIntegrityError, match="absent"):
        lock.verify("b.csv", record(b"b"))


def test_invalid_sha_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FileRecord(sha256="pas-un-sha", size_bytes=1)


def test_manifest_carries_source_and_version() -> None:
    files = {"a.csv": record(b"a")}
    manifest = RawManifest.build("github:x/y", "3" * 40, datetime(2026, 1, 1, tzinfo=UTC), files)
    assert manifest.dataset_version == DataLock.build({}, files).dataset_version
