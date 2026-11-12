"""Envoi d'une carte au webhook Teams (Workflows). Ne lève jamais d'exception et n'écrit jamais l'URL.

L'URL du webhook est un secret : elle vient de `TENNIS_TEAMS_WEBHOOK_URL` (`.env`, ignoré par Git) et n'apparaît ni
dans les logs, ni dans les messages d'erreur, ni dans les tags. Une alerte ne décide jamais rien : un envoi raté est
signalé, il ne fait pas échouer la chaîne.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import httpx
from pydantic import SecretStr

from tennis_mlops.alerts.cards import Alert, teams_message


class Delivery(StrEnum):
    """Résultat d'un envoi."""

    SENT = "sent"  # Teams a accepté la carte (Workflows répond 202)
    DISABLED = "disabled"  # aucun webhook configuré : rien n'est envoyé, ce n'est pas une erreur
    FAILED = "failed"  # refus ou réseau : la raison est dans `detail`, sans l'URL


@dataclass(frozen=True)
class DeliveryResult:
    """Ce qu'est devenue une carte."""

    status: Delivery
    detail: str = ""


class TeamsNotifier:
    """Poste des cartes Adaptive au webhook Workflows d'un canal."""

    def __init__(
        self, webhook_url: SecretStr | None, timeout_s: float = 10.0, client: httpx.Client | None = None
    ) -> None:
        url = webhook_url.get_secret_value().strip() if webhook_url else ""
        self._url = url
        self._timeout = timeout_s
        self._client = client

    @property
    def enabled(self) -> bool:
        """Vrai si un webhook est configuré."""
        return bool(self._url)

    def send(self, alert: Alert) -> DeliveryResult:
        """Poste la carte ; le résultat dit si Teams l'a acceptée."""
        if not self.enabled:
            return DeliveryResult(Delivery.DISABLED, "TENNIS_TEAMS_WEBHOOK_URL absent : carte non envoyée")
        try:
            if self._client is not None:
                response = self._client.post(self._url, json=teams_message(alert), timeout=self._timeout)
            else:
                response = httpx.post(self._url, json=teams_message(alert), timeout=self._timeout)
        except httpx.HTTPError as exc:  # le message d'httpx contient l'URL : on ne garde que le type
            return DeliveryResult(Delivery.FAILED, f"webhook injoignable ({type(exc).__name__})")
        if response.status_code >= 300:
            return DeliveryResult(Delivery.FAILED, f"webhook : HTTP {response.status_code} {response.text[:200]}")
        return DeliveryResult(Delivery.SENT, f"HTTP {response.status_code}")
