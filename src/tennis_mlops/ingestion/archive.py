"""Zone brute locale : ce que la source a réellement envoyé, gardé tel quel avec son verdict.

```
state/lots/<lot_id>/payload.json       lot accepté : réponse brute de la source
state/lots/<lot_id>/validation.json    verdict des contrôles
state/quarantine/<lot_id>/…            lot refusé : mêmes fichiers, pour analyse et rejeu
```

Garder le brut permet de rejouer un lot sans rappeler la source et de montrer, pendant une démo, la donnée exacte
qui a été bloquée.
"""

from __future__ import annotations

import json
from pathlib import Path

from tennis_mlops.ingestion.lots import Lot
from tennis_mlops.validation.validator import ValidationReport

LOTS_ZONE = "lots"
QUARANTINE_ZONE = "quarantine"


class LotArchive:
    """Écrit les lots et leur verdict sur disque."""

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir

    def save(self, lot: Lot, report: ValidationReport) -> Path:
        """Archive le lot dans la zone qui correspond à son verdict ; renvoie son dossier."""
        zone = LOTS_ZONE if report.success else QUARANTINE_ZONE
        folder = self._state_dir / zone / lot.lot_id
        folder.mkdir(parents=True, exist_ok=True)
        envelope = {
            "endpoint": lot.payload.endpoint,
            "simulated_now": lot.payload.simulated_now.isoformat(),
            "dataset_version": lot.payload.dataset_version,
            "fetched_at": lot.payload.fetched_at.isoformat(),
            "data": lot.payload.data,
        }
        (folder / "payload.json").write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        (folder / "validation.json").write_text(report.to_json(), encoding="utf-8")
        return folder
