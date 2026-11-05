"""Doublures de test : sources de données en mémoire, sans réseau."""

from __future__ import annotations

from pathlib import Path

from tennis_mlops.data.errors import SourceUnavailableError


class InMemorySource:
    """Source qui sert des contenus fixés en mémoire et compte les téléchargements."""

    def __init__(
        self,
        name: str,
        contents: dict[str, bytes],
        revision: str = "a" * 40,
        available: bool = True,
    ) -> None:
        self._name = name
        self.contents = contents
        self.revision = revision
        self.available = available
        self.fetched: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return self._name

    def resolve_revision(self) -> str:
        if not self.available:
            raise SourceUnavailableError(f"{self._name} hors ligne")
        return self.revision

    def fetch(self, filename: str, revision: str, destination: Path) -> None:
        if not self.available:
            raise SourceUnavailableError(f"{self._name} hors ligne")
        self.fetched.append((filename, revision))
        destination.write_bytes(self.contents[filename])
