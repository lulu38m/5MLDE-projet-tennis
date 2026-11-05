"""Ligne de commande des données : `python -m tennis_mlops.data {lock,download,verify,profile}`.

Codes de sortie : 0 succès · 1 intégrité (contenu inattendu) · 2 aucune source joignable.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig
from tennis_mlops.data.catalog import DatasetCatalog
from tennis_mlops.data.download import RawDataDownloader
from tennis_mlops.data.errors import DataIntegrityError, SourceUnavailableError
from tennis_mlops.data.lockfile import DataLock
from tennis_mlops.data.profile import RawDataProfiler
from tennis_mlops.data.sources import DataSource, build_source
from tennis_mlops.log import configure_logging

logger = logging.getLogger("tennis_mlops.data")

EXIT_OK, EXIT_INTEGRITY, EXIT_UNAVAILABLE = 0, 1, 2
SourceFactory = Callable[[ProjectConfig], list[DataSource]]


def default_sources(config: ProjectConfig) -> list[DataSource]:
    """Instancie les sources dans l'ordre de la configuration."""
    return [build_source(source) for source in config.data.sources]


class DataCommands:
    """Une méthode par sous-commande ; chacune renvoie un code de sortie."""

    def __init__(self, config: ProjectConfig, root: Path, sources: Sequence[DataSource]) -> None:
        self._config = config
        self._lock_path = config.data.resolved_lock_file(root)
        self._raw_dir = config.data.resolved_raw_dir(root)
        self._catalog = DatasetCatalog.from_config(config.data)
        self._sources = sources

    def _downloader(self, lock: DataLock | None) -> RawDataDownloader:
        return RawDataDownloader(self._catalog, self._sources, self._raw_dir, lock)

    def _load_lock(self) -> DataLock | None:
        return DataLock.load(self._lock_path) if self._lock_path.is_file() else None

    def lock(self) -> int:
        """(Re)génère le verrou à partir des révisions actuelles des sources."""
        downloader = self._downloader(lock=None)
        report = downloader.download()
        lock = downloader.build_lock(report.manifest)
        lock.save(self._lock_path)
        logger.info("Verrou écrit : %s (dataset_version=%s)", self._lock_path, lock.dataset_version)
        return EXIT_OK

    def download(self) -> int:
        """Télécharge (ou réutilise) les fichiers et les vérifie contre le verrou."""
        lock = self._load_lock()
        if lock is None:
            logger.warning("Aucun verrou trouvé : téléchargement non vérifié. Lancer `make data-lock`.")
        report = self._downloader(lock).download()
        if report.failed_sources:
            logger.warning("Mode dégradé : sources indisponibles %s", ", ".join(report.failed_sources))
        logger.info("OK — dataset_version=%s source=%s", report.manifest.dataset_version, report.manifest.source)
        return EXIT_OK

    def verify(self) -> int:
        """Vérifie les fichiers locaux contre le verrou."""
        manifest = self._downloader(self._load_lock()).verify()
        logger.info("OK — %d fichiers conformes, dataset_version=%s", len(manifest.files), manifest.dataset_version)
        return EXIT_OK

    def profile(self) -> int:
        """Vérifie puis profile les données brutes."""
        manifest = self._downloader(self._load_lock()).verify()
        profile = RawDataProfiler(self._catalog, self._raw_dir).profile(manifest.dataset_version)
        output = self._raw_dir / "profile.json"
        profile.save(output)
        for year in profile.years:
            logger.info("%d : %5d matchs, %3d tournois, %s", year.year, year.matches, year.tournaments, year.levels)
        logger.info("Issues des matchs : %s", profile.statuses)
        logger.info("Profil écrit : %s", output)
        return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Analyseur des arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.data", description=__doc__)
    parser.add_argument("command", choices=["lock", "download", "verify", "profile"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main(
    argv: Sequence[str] | None = None,
    root: Path | None = None,
    source_factory: SourceFactory = default_sources,
) -> int:
    """Point d'entrée ; renvoie le code de sortie."""
    configure_logging()
    args = build_parser().parse_args(argv)
    config = ProjectConfig.from_yaml(args.config)
    commands = DataCommands(config, root or args.config.resolve().parents[1], source_factory(config))
    action: Callable[[], int] = getattr(commands, args.command)
    try:
        return action()
    except DataIntegrityError as exc:
        logger.error("Intégrité des données KO : %s", exc)
        return EXIT_INTEGRITY
    except SourceUnavailableError as exc:
        logger.error("Données indisponibles : %s", exc)
        return EXIT_UNAVAILABLE


if __name__ == "__main__":
    raise SystemExit(main())
