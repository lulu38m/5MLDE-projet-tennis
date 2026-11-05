"""Liste exacte des fichiers bruts nécessaires, déduite de la configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from tennis_mlops.config import FIRST_ATP_YEAR, DataConfig


class FileKind(StrEnum):
    """Nature d'un fichier brut."""

    MATCHES = "matches"
    RANKINGS = "rankings"
    PLAYERS = "players"
    DOCUMENTATION = "documentation"


@dataclass(frozen=True, slots=True)
class RawFile:
    """Un fichier du dataset Sackmann, tel que nommé dans le dossier `atp/` de la source."""

    name: str
    kind: FileKind
    year: int | None = None


class DatasetCatalog:
    """Calcule les fichiers à télécharger pour une plage d'années de matchs.

    Les classements Sackmann sont découpés par décennie (`atp_rankings_10s.csv`…) ; l'année en cours est dans
    `atp_rankings_current.csv`, toujours inclus.
    """

    PLAYERS_FILE = "atp_players.csv"
    CURRENT_RANKINGS_FILE = "atp_rankings_current.csv"
    DOCUMENTATION_FILES = ("matches_data_dictionary.txt", "UPSTREAM_README.md")

    def __init__(self, first_year: int, last_year: int) -> None:
        if first_year < FIRST_ATP_YEAR or first_year > last_year:
            msg = f"plage d'années invalide : {first_year}-{last_year}"
            raise ValueError(msg)
        self._first_year = first_year
        self._last_year = last_year

    @classmethod
    def from_config(cls, config: DataConfig) -> Self:
        """Construit le catalogue à partir de la section `data` de la configuration."""
        return cls(config.first_match_year, config.last_match_year)

    @property
    def years(self) -> range:
        """Années de matchs couvertes."""
        return range(self._first_year, self._last_year + 1)

    def files(self) -> tuple[RawFile, ...]:
        """Tous les fichiers nécessaires, dans un ordre stable."""
        matches = [RawFile(name, FileKind.MATCHES, year) for year, name in self.match_files().items()]
        decades = sorted({year // 10 * 10 for year in self.years})
        rankings = [RawFile(f"atp_rankings_{decade % 100:02d}s.csv", FileKind.RANKINGS) for decade in decades]
        rankings.append(RawFile(self.CURRENT_RANKINGS_FILE, FileKind.RANKINGS))
        players = [RawFile(self.PLAYERS_FILE, FileKind.PLAYERS)]
        docs = [RawFile(name, FileKind.DOCUMENTATION) for name in self.DOCUMENTATION_FILES]
        return (*matches, *rankings, *players, *docs)

    def names(self) -> tuple[str, ...]:
        """Noms des fichiers nécessaires."""
        return tuple(raw_file.name for raw_file in self.files())

    def match_files(self) -> dict[int, str]:
        """Nom du fichier de matchs pour chaque année."""
        return {year: f"atp_matches_{year}.csv" for year in self.years}

    def of_kind(self, kind: FileKind) -> tuple[RawFile, ...]:
        """Fichiers d'une nature donnée."""
        return tuple(raw_file for raw_file in self.files() if raw_file.kind is kind)
