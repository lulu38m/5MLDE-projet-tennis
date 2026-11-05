"""Profilage rapide des données brutes, pour objectiver les choix de la suite (fenêtre, contrôles, features).

Le résultat est un objet typé, écrit en JSON dans `data/raw/profile.json` et réutilisable plus tard comme
référence (par exemple pour les contrôles de distribution Great Expectations).
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from tennis_mlops.data.catalog import DatasetCatalog, FileKind
from tennis_mlops.data.score import classify_score

KEY_COLUMNS = (
    "surface",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "winner_age",
    "loser_age",
    "winner_ht",
    "loser_ht",
    "minutes",
    "w_ace",
)


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class YearProfile(_Report):
    """Vue d'une année de matchs."""

    year: int
    matches: int
    tournaments: int
    first_tourney_date: str
    last_tourney_date: str
    levels: dict[str, int]
    surfaces: dict[str, int]


class RankingsProfile(_Report):
    """Vue des fichiers de classement."""

    rows: int
    weeks: int
    first_date: str
    last_date: str


class DataProfile(_Report):
    """Profil complet du dataset brut."""

    dataset_version: str
    generated_at: datetime
    total_matches: int
    duplicate_match_keys: int
    years: list[YearProfile]
    null_rate_pct: dict[str, float]
    statuses: dict[str, int]
    players: int
    rankings: RankingsProfile

    def save(self, path: Path) -> None:
        """Écrit le profil en JSON."""
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


class RawDataProfiler:
    """Calcule le profil des fichiers bruts d'un catalogue."""

    def __init__(self, catalog: DatasetCatalog, raw_dir: Path) -> None:
        self._catalog = catalog
        self._raw_dir = raw_dir

    def load_matches(self) -> pd.DataFrame:
        """Concatène les fichiers de matchs (types fixés pour les identifiants)."""
        frames = [
            pd.read_csv(
                self._raw_dir / name,
                dtype={"tourney_id": "string", "score": "string", "surface": "string"},
                low_memory=False,
            ).assign(file_year=year)
            for year, name in self._catalog.match_files().items()
        ]
        return pd.concat(frames, ignore_index=True)

    def profile(self, dataset_version: str) -> DataProfile:
        """Profil des matchs, joueurs et classements."""
        matches = self.load_matches()
        return DataProfile(
            dataset_version=dataset_version,
            generated_at=datetime.now(UTC),
            total_matches=len(matches),
            duplicate_match_keys=int(matches.duplicated(["tourney_id", "match_num"]).sum()),
            years=[self._year_profile(year, group) for year, group in matches.groupby("file_year")],
            null_rate_pct={
                column: round(float(matches[column].isna().mean()) * 100, 2)
                for column in KEY_COLUMNS
                if column in matches
            },
            statuses=dict(Counter(classify_score(None if pd.isna(s) else str(s)).value for s in matches["score"])),
            players=len(pd.read_csv(self._raw_dir / DatasetCatalog.PLAYERS_FILE, usecols=["player_id"])),
            rankings=self._rankings_profile(),
        )

    @staticmethod
    def _year_profile(year: object, group: pd.DataFrame) -> YearProfile:
        dates = group["tourney_date"].astype(int)
        return YearProfile(
            year=int(str(year)),
            matches=len(group),
            tournaments=int(group["tourney_id"].nunique()),
            first_tourney_date=str(dates.min()),
            last_tourney_date=str(dates.max()),
            levels={str(k): int(v) for k, v in group["tourney_level"].value_counts().items()},
            surfaces={str(k): int(v) for k, v in group["surface"].fillna("unknown").value_counts().items()},
        )

    def _rankings_profile(self) -> RankingsProfile:
        dates = pd.concat(
            [
                pd.read_csv(self._raw_dir / raw_file.name, usecols=["ranking_date"])["ranking_date"]
                for raw_file in self._catalog.of_kind(FileKind.RANKINGS)
            ],
            ignore_index=True,
        )
        return RankingsProfile(
            rows=len(dates),
            weeks=int(dates.nunique()),
            first_date=str(dates.min()),
            last_date=str(dates.max()),
        )
