"""Sources du dataset : Hugging Face (principale) et miroir GitHub (secours).

Chaque source sait faire deux choses : résoudre une révision mobile (`main`) en sha immuable, et télécharger un
fichier à cette révision. Toute erreur réseau est convertie en `SourceUnavailableError`, ce qui permet au
téléchargeur de basculer sur la source suivante.

Les accès réseau passent par des dépendances injectables (`HubBackend`, `CommandRunner`, `httpx.Client`) : les
tests n'ont jamais besoin d'Internet.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import httpx
import httpx2
from huggingface_hub import HfApi, hf_hub_download

from tennis_mlops.config import SourceConfig, SourceKind
from tennis_mlops.data.errors import SourceUnavailableError

logger = logging.getLogger(__name__)

_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class DataSource(Protocol):
    """Contrat commun à toutes les sources."""

    @property
    def name(self) -> str:
        """Identifiant stable (clé dans le fichier de verrouillage)."""
        ...

    def resolve_revision(self) -> str:
        """Résout la révision configurée en sha immuable."""
        ...

    def fetch(self, filename: str, revision: str, destination: Path) -> None:
        """Télécharge `filename` à la révision donnée vers `destination`."""
        ...


# --------------------------------------------------------------------------- Hugging Face


class HubBackend(Protocol):
    """Opérations Hugging Face utilisées par le projet."""

    def dataset_sha(self, repo: str, revision: str, timeout_s: float) -> str:
        """Sha du commit correspondant à `revision`."""
        ...

    def download(self, repo: str, path_in_repo: str, revision: str, timeout_s: float) -> Path:
        """Télécharge un fichier dans le cache local et renvoie son chemin."""
        ...


class HuggingFaceHubBackend:
    """Implémentation réelle, basée sur `huggingface_hub` (aucun token nécessaire : dataset public)."""

    def __init__(self, api: HfApi | None = None) -> None:
        self._api = api or HfApi()

    def dataset_sha(self, repo: str, revision: str, timeout_s: float) -> str:
        """Sha du commit correspondant à `revision`."""
        info = self._api.dataset_info(repo, revision=revision, timeout=timeout_s)
        if info.sha is None:
            msg = f"Hugging Face n'a pas renvoyé de sha pour {repo}@{revision}"
            raise SourceUnavailableError(msg)
        return info.sha

    def download(self, repo: str, path_in_repo: str, revision: str, timeout_s: float) -> Path:
        """Télécharge un fichier dans le cache Hugging Face et renvoie son chemin."""
        local = hf_hub_download(repo, path_in_repo, repo_type="dataset", revision=revision, etag_timeout=timeout_s)
        return Path(str(local))


class HuggingFaceSource:
    """Source principale : le dataset Hugging Face choisi pour le projet."""

    # huggingface_hub 2.x s'appuie sur `httpx2` (pas `httpx`) : toutes ses erreurs HTTP, y compris
    # HfHubHTTPError, héritent de httpx2.HTTPError. OSError couvre le cache local et le mode hors ligne.
    _NETWORK_ERRORS = (httpx2.HTTPError, OSError)

    def __init__(self, config: SourceConfig, backend: HubBackend | None = None) -> None:
        self._config = config
        self._backend = backend or HuggingFaceHubBackend()

    @property
    def name(self) -> str:
        """Identifiant stable de la source."""
        return self._config.name

    def resolve_revision(self) -> str:
        """Résout la révision configurée en sha de commit Hugging Face."""
        try:
            return self._backend.dataset_sha(self._config.repo, self._config.revision, self._config.timeout_s)
        except self._NETWORK_ERRORS as exc:
            msg = f"{self.name} injoignable ({type(exc).__name__}: {exc})"
            raise SourceUnavailableError(msg) from exc

    def fetch(self, filename: str, revision: str, destination: Path) -> None:
        """Télécharge le fichier via le cache Hugging Face puis le copie vers `destination`."""
        try:
            cached = self._backend.download(
                self._config.repo, f"{self._config.subdir}/{filename}", revision, self._config.timeout_s
            )
            shutil.copyfile(cached, destination)
        except self._NETWORK_ERRORS as exc:
            msg = f"{self.name} : échec de téléchargement de {filename} ({type(exc).__name__}: {exc})"
            raise SourceUnavailableError(msg) from exc


# --------------------------------------------------------------------------- GitHub


CommandRunner = Callable[[Sequence[str], float], str]


def run_command(command: Sequence[str], timeout_s: float) -> str:
    """Exécute une commande et renvoie sa sortie standard (lève en cas d'échec)."""
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=True)
    return completed.stdout


class GitHubSource:
    """Source de secours : miroir GitHub du même dataset, lu via `raw.githubusercontent.com`."""

    RAW_URL = "https://raw.githubusercontent.com/{repo}/{revision}/{subdir}/{filename}"

    def __init__(
        self,
        config: SourceConfig,
        client: httpx.Client | None = None,
        runner: CommandRunner = run_command,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=config.timeout_s, follow_redirects=True)
        self._runner = runner

    @property
    def name(self) -> str:
        """Identifiant stable de la source."""
        return self._config.name

    def resolve_revision(self) -> str:
        """Résout la révision avec `git ls-remote` (pas de quota d'API, rien à cloner)."""
        revision = self._config.revision
        if _SHA_PATTERN.match(revision):
            return revision
        url = f"https://github.com/{self._config.repo}.git"
        try:
            output = self._runner(["git", "ls-remote", url, revision], self._config.timeout_s)
        except (subprocess.SubprocessError, OSError) as exc:
            msg = f"{self.name} injoignable ({type(exc).__name__})"
            raise SourceUnavailableError(msg) from exc
        for line in output.splitlines():
            sha, _, ref = line.partition("\t")
            if ref in {revision, f"refs/heads/{revision}", f"refs/tags/{revision}"} and _SHA_PATTERN.match(sha):
                return sha
        msg = f"{self.name} : révision « {revision} » introuvable"
        raise SourceUnavailableError(msg)

    def fetch(self, filename: str, revision: str, destination: Path) -> None:
        """Télécharge le fichier en streaming vers `destination`."""
        url = self.RAW_URL.format(
            repo=self._config.repo, revision=revision, subdir=self._config.subdir, filename=filename
        )
        try:
            with self._client.stream("GET", url) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
        except httpx.HTTPError as exc:
            msg = f"{self.name} : échec de téléchargement de {filename} ({type(exc).__name__})"
            raise SourceUnavailableError(msg) from exc


def build_source(config: SourceConfig) -> DataSource:
    """Instancie la source correspondant à la configuration."""
    match config.kind:
        case SourceKind.HUGGINGFACE:
            return HuggingFaceSource(config)
        case SourceKind.GITHUB:
            return GitHubSource(config)
