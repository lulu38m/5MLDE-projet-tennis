"""Les matchs bruts de Sackmann, dans le format de la table `matches`, avec les statistiques de service.

On réutilise la préparation de `mock_source` (date estimée de chaque match dans son tournoi, statut déduit du
score, identifiant `tourney_id-match_num`) et on y rattache les colonnes de statistiques (`w_ace`, `l_svpt`…) que
le pipeline n'utilise pas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mock_source.repository import prepare_matches
from tennis_mlops.data.catalog import DatasetCatalog

RAW_DIR = Path("data/raw")
STATS = ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced")


def load_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Tous les matchs des fichiers verrouillés, triés par date estimée (une ligne par match)."""
    names = sorted(p.name for p in raw_dir.glob("atp_matches_*.csv"))
    if not names:
        msg = f"aucun fichier atp_matches_*.csv dans {raw_dir} : lancer `make data`"
        raise FileNotFoundError(msg)
    raw = pd.concat(
        [pd.read_csv(raw_dir / n, dtype={"tourney_id": "string", "score": "string"}, low_memory=False) for n in names],
        ignore_index=True,
    )
    raw = raw.dropna(subset=["winner_id", "loser_id", "match_num", "tourney_date"])
    raw["match_id"] = raw["tourney_id"].astype(str) + "-" + raw["match_num"].astype(int).astype(str)
    raw = raw.drop_duplicates("match_id")
    matches = prepare_matches(raw, first_round_notice_days=2)
    stats = raw[["match_id", *[f"{s}_{c}" for s in ("w", "l") for c in STATS]]]
    matches = matches.merge(stats, on="match_id", how="left", validate="one_to_one")
    matches["match_date"] = pd.to_datetime(matches["match_date"]).dt.date
    return matches.sort_values(["match_date", "tourney_id", "match_num"], kind="stable", ignore_index=True)


def players(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Fiche des joueurs (date de naissance, main, taille, pays)."""
    return pd.read_csv(raw_dir / DatasetCatalog.PLAYERS_FILE, dtype={"dob": "string"}, low_memory=False)
