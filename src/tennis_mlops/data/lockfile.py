"""Verrouillage du dataset : révisions résolues et empreinte SHA-256 de chaque fichier.

Même principe que `uv.lock` pour les dépendances : `configs/data.lock.json` est versionné, et tout téléchargement
est vérifié contre lui. Deux machines qui passent la vérification ont des données identiques à l'octet près.

`dataset_version` est dérivé **du contenu seul** (pas de la source) : un même fichier téléchargé depuis Hugging Face
ou depuis le miroir GitHub donne la même version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tennis_mlops.data.errors import DataIntegrityError

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Empreinte SHA-256 d'un fichier, lue par blocs pour ne pas charger les gros CSV en mémoire."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_version(files: Mapping[str, FileRecord]) -> str:
    """Version courte du dataset, indépendante de l'ordre des fichiers et de la source."""
    canonical = "\n".join(f"{name}:{files[name].sha256}" for name in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class FileRecord(BaseModel):
    """Empreinte d'un fichier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @classmethod
    def of(cls, path: Path) -> Self:
        """Calcule l'empreinte d'un fichier local."""
        return cls(sha256=sha256_file(path), size_bytes=path.stat().st_size)


class _FileSet(BaseModel):
    """Ensemble de fichiers dont la version est recalculée et contrôlée au chargement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    files: dict[str, FileRecord]
    dataset_version: str

    @model_validator(mode="after")
    def _check_dataset_version(self) -> Self:
        expected = compute_dataset_version(self.files)
        if self.dataset_version != expected:
            msg = f"dataset_version incohérent ({self.dataset_version} ≠ {expected}) : fichier modifié à la main ?"
            raise ValueError(msg)
        return self

    def save(self, path: Path) -> None:
        """Écrit le JSON indenté, métadonnées en tête et fichiers en dernier, pour des diffs Git lisibles."""
        data = self.model_dump(mode="json")
        files = data.pop("files")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**data, "files": files}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Relit et valide un fichier JSON."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class DataLock(_FileSet):
    """Contenu attendu du dataset (versionné dans Git)."""

    schema_version: Literal[1] = 1
    revisions: dict[str, str] = Field(description="Révision résolue (sha) par source")

    @classmethod
    def build(cls, revisions: Mapping[str, str], files: Mapping[str, FileRecord]) -> Self:
        """Construit un verrou en calculant sa version."""
        ordered = {name: files[name] for name in sorted(files)}
        return cls(
            revisions=dict(sorted(revisions.items())),
            files=ordered,
            dataset_version=compute_dataset_version(ordered),
        )

    def verify(self, name: str, actual: FileRecord) -> None:
        """Lève `DataIntegrityError` si un fichier ne correspond pas au verrou."""
        expected = self.files.get(name)
        if expected is None:
            msg = f"{name} absent du fichier de verrouillage : relancer `make data-lock`"
            raise DataIntegrityError(msg)
        if expected.sha256 != actual.sha256:
            msg = f"{name} : SHA-256 {actual.sha256[:12]}… ≠ attendu {expected.sha256[:12]}…"
            raise DataIntegrityError(msg)


class RawManifest(_FileSet):
    """Trace locale d'un téléchargement (dans `data/raw/manifest.json`, non versionné)."""

    source: str
    revision: str
    downloaded_at: datetime

    @classmethod
    def build(cls, source: str, revision: str, downloaded_at: datetime, files: Mapping[str, FileRecord]) -> Self:
        """Construit un manifeste en calculant sa version."""
        ordered = {name: files[name] for name in sorted(files)}
        return cls(
            source=source,
            revision=revision,
            downloaded_at=downloaded_at,
            files=ordered,
            dataset_version=compute_dataset_version(ordered),
        )
