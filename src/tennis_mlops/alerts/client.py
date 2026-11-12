"""Client du service d'alertes, pour le flow d'entraînement et la ligne de commande du registre.

Best effort : si `alert_sink` ne répond pas (stack arrêtée, test local), la décision du registre reste valable ; on le
signale en une ligne, sans faire échouer l'opération.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from tennis_mlops.alerts.builders import RegistryNotice

TIMEOUT_S = 3.0


@dataclass(frozen=True)
class Notification:
    """Ce qu'a répondu le service (ou pourquoi il n'a pas répondu)."""

    delivered: bool  # le service a reçu la décision (qu'il l'envoie sur Teams ou non)
    status: str  # sent, filtered, disabled, failed, duplicate, ignored, unreachable
    detail: str = ""

    def line(self) -> str:
        """Une ligne pour le terminal."""
        labels = {
            "sent": "carte envoyée sur Teams",
            "filtered": "gardée dans l'historique (type absent de teams_events)",
            "disabled": "gardée dans l'historique (aucun webhook Teams)",
            "duplicate": "déjà notifiée",
            "ignored": "pas d'alerte pour cette décision",
        }
        if self.status == "unreachable" or not self.delivered:
            return f"alerte non transmise : {self.detail}"
        if self.status == "failed":
            return f"alerte : envoi Teams en échec ({self.detail})"
        return f"alerte : {labels.get(self.status, self.status)}"


class AlertSinkClient:
    """POST /registry vers `alert_sink`."""

    def __init__(self, base_url: str, timeout_s: float = TIMEOUT_S, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = client

    def notify_registry(self, notice: RegistryNotice) -> Notification:
        """Transmet une décision du registre ; ne lève jamais."""
        url, body = f"{self.base_url}/registry", notice.model_dump(mode="json")
        try:
            if self._client is not None:
                response = self._client.post(url, json=body, timeout=self.timeout_s)
            else:
                response = httpx.post(url, json=body, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            return Notification(False, "unreachable", f"service d'alertes injoignable ({type(exc).__name__})")
        if response.status_code >= 300:
            return Notification(False, "failed", f"service d'alertes : HTTP {response.status_code}")
        data = response.json()
        return Notification(True, str(data.get("status", "?")), str(data.get("detail", "")))
