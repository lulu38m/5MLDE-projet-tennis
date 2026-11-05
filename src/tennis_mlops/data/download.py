"""Téléchargement reproductible des fichiers bruts, avec vérification et fallback entre sources.

Règles :

- une source **injoignable** → on passe à la suivante (fallback), avec un avertissement dans les logs ;
- un contenu **différent** du verrou → arrêt immédiat (`DataIntegrityError`), quelle que soit la source ;
- un fichier local déjà conforme au verrou n'est pas re-téléchargé (cache) ;
- un fichier n'est déplacé à sa place définitive qu'une fois vérifié (jamais de fichier à moitié écrit) ;
- tout fichier est rendu lisible par tous (montage en lecture seule dans les conteneurs non-root).
"""

from __future__ import annotations

import logging
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.errors import DataIntegrityError, SourceUnavailableError
from tennis_mlops.data.lockfile import DataLock, FileRecord, RawManifest
from tennis_mlops.data.sources import DataSource

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
READ_FOR_ALL = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


@dataclass(frozen=True, slots=True)
class DownloadReport:
    """Résumé d'un téléchargement."""

    manifest: RawManifest
    downloaded: tuple[str, ...]
    reused: tuple[str, ...]
    failed_sources: tuple[str, ...]


class RawDataDownloader:
    """Télécharge le catalogue depuis la première source disponible et le vérifie contre le verrou."""

    def __init__(
        self,
        catalog: DatasetCatalog,
        sources: Sequence[DataSource],
        raw_dir: Path,
        lock: DataLock | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not sources:
            msg = "au moins une source est nécessaire"
            raise ValueError(msg)
        self._catalog = catalog
        self._sources = tuple(sources)
        self._raw_dir = raw_dir
        self._lock = lock
        self._clock = clock

    @property
    def manifest_path(self) -> Path:
        """Emplacement du manifeste local."""
        return self._raw_dir / MANIFEST_NAME

    def download(self) -> DownloadReport:
        """Télécharge depuis la première source joignable, dans l'ordre de la configuration."""
        failures: list[str] = []
        for source in self._sources:
            try:
                report = self._download_from(source)
            except SourceUnavailableError as exc:
                logger.warning("Source indisponible, bascule sur la suivante : %s", exc)
                failures.append(source.name)
                continue
            return DownloadReport(report.manifest, report.downloaded, report.reused, tuple(failures))
        msg = f"aucune source disponible ({', '.join(failures)})"
        raise SourceUnavailableError(msg)

    def build_lock(self, manifest: RawManifest) -> DataLock:
        """Construit un verrou à partir d'un téléchargement, en résolvant la révision de chaque source joignable.

        Les sources injoignables sont omises : leur révision sera ajoutée au prochain `make data-lock`.
        """
        revisions = {manifest.source: manifest.revision}
        for source in self._sources:
            if source.name in revisions:
                continue
            try:
                revisions[source.name] = source.resolve_revision()
            except SourceUnavailableError as exc:
                logger.warning("Révision non verrouillée pour %s : %s", source.name, exc)
        return DataLock.build(revisions, manifest.files)

    def verify(self) -> RawManifest:
        """Recalcule l'empreinte de chaque fichier local et la compare au verrou."""
        if self._lock is None:
            msg = "pas de fichier de verrouillage : lancer `make data-lock`"
            raise DataIntegrityError(msg)
        self._lock.verify_directory(self._raw_dir, self._catalog.names())
        return RawManifest.load(self.manifest_path)

    def _download_from(self, source: DataSource) -> DownloadReport:
        revision = self._pinned_revision(source) or source.resolve_revision()
        logger.info("Téléchargement depuis %s @ %s", source.name, revision[:12])
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        records: dict[str, FileRecord] = {}
        downloaded: list[str] = []
        reused: list[str] = []
        for name in self._catalog.names():
            target = self._raw_dir / name
            cached = self._valid_local_record(name, target)
            if cached is not None:
                records[name] = cached
                reused.append(name)
                continue
            records[name] = self._fetch_verified(source, name, revision, target)
            downloaded.append(name)
        manifest = RawManifest.build(source.name, revision, self._clock(), records)
        manifest.save(self.manifest_path)
        logger.info(
            "Dataset %s : %d fichiers téléchargés, %d réutilisés",
            manifest.dataset_version,
            len(downloaded),
            len(reused),
        )
        return DownloadReport(manifest, tuple(downloaded), tuple(reused), ())

    def _pinned_revision(self, source: DataSource) -> str | None:
        return self._lock.revisions.get(source.name) if self._lock else None

    def _valid_local_record(self, name: str, target: Path) -> FileRecord | None:
        if self._lock is None or not target.is_file():
            return None
        record = FileRecord.of(target)
        expected = self._lock.files.get(name)
        if expected is None or expected.sha256 != record.sha256:
            return None
        make_world_readable(target)
        return record

    def _fetch_verified(self, source: DataSource, name: str, revision: str, target: Path) -> FileRecord:
        partial = target.with_name(target.name + ".part")
        try:
            source.fetch(name, revision, partial)
            record = FileRecord.of(partial)
            if self._lock is not None:
                self._lock.verify(name, record)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        partial.replace(target)
        make_world_readable(target)
        return record


def make_world_readable(path: Path) -> None:
    """Ajoute la lecture pour tous : les conteneurs (utilisateur non-root, uid 10001) lisent `data/raw` en montage.

    Les données sont publiques (licence CC BY-NC-SA) : aucune raison de les réserver à leur propriétaire. Certains
    outils créent leurs fichiers en mode 600, ce qui fait échouer le démarrage de `mock_source` dans Docker.
    """
    mode = path.stat().st_mode
    if mode & READ_FOR_ALL != READ_FOR_ALL:
        path.chmod(mode | READ_FOR_ALL)
