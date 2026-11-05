import subprocess
from collections.abc import Sequence
from pathlib import Path

import httpx
import httpx2
import pytest

from tennis_mlops.config import SourceConfig, SourceKind
from tennis_mlops.data.errors import SourceUnavailableError
from tennis_mlops.data.sources import GitHubSource, HuggingFaceSource, build_source

SHA = "8" * 40
HF = SourceConfig(kind=SourceKind.HUGGINGFACE, repo="org/data")
GH = SourceConfig(kind=SourceKind.GITHUB, repo="org/mirror")


class FakeHub:
    def __init__(self, tmp_path: Path, error: Exception | None = None) -> None:
        self.tmp_path = tmp_path
        self.error = error
        self.calls: list[tuple[str, str, str]] = []

    def dataset_sha(self, repo: str, revision: str, timeout_s: float) -> str:
        if self.error:
            raise self.error
        return SHA

    def download(self, repo: str, path_in_repo: str, revision: str, timeout_s: float) -> Path:
        if self.error:
            raise self.error
        self.calls.append((repo, path_in_repo, revision))
        cached = self.tmp_path / "cache.csv"
        cached.write_text("hf-content")
        return cached


def test_huggingface_resolves_and_downloads_from_subdir(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path)
    source = HuggingFaceSource(HF, backend=hub)
    assert source.name == "huggingface:org/data"
    assert source.resolve_revision() == SHA
    source.fetch("atp_players.csv", SHA, tmp_path / "out.csv")
    assert hub.calls == [("org/data", "atp/atp_players.csv", SHA)]
    assert (tmp_path / "out.csv").read_text() == "hf-content"


@pytest.mark.parametrize("error", [httpx2.ProxyError("403 Forbidden"), OSError("disque")])
def test_huggingface_network_errors_become_unavailable(tmp_path: Path, error: Exception) -> None:
    source = HuggingFaceSource(HF, backend=FakeHub(tmp_path, error))
    with pytest.raises(SourceUnavailableError, match="injoignable"):
        source.resolve_revision()
    with pytest.raises(SourceUnavailableError, match=r"atp_players\.csv"):
        source.fetch("atp_players.csv", SHA, tmp_path / "out.csv")


def test_github_resolves_branch_with_ls_remote() -> None:
    source = GitHubSource(GH, runner=lambda cmd, t: f"{SHA}\trefs/heads/main\n{'9' * 40}\trefs/heads/mainline\n")
    assert source.name == "github:org/mirror"
    assert source.resolve_revision() == SHA


def test_github_keeps_an_explicit_sha_without_network() -> None:
    def fail(command: Sequence[str], timeout_s: float) -> str:
        raise AssertionError("aucun appel réseau attendu")

    pinned = SourceConfig(kind=SourceKind.GITHUB, repo="org/mirror", revision=SHA)
    assert GitHubSource(pinned, runner=fail).resolve_revision() == SHA


def test_github_unknown_revision_or_git_failure() -> None:
    with pytest.raises(SourceUnavailableError, match="introuvable"):
        GitHubSource(GH, runner=lambda cmd, t: "").resolve_revision()

    def boom(command: Sequence[str], timeout_s: float) -> str:
        raise subprocess.CalledProcessError(128, command)

    with pytest.raises(SourceUnavailableError, match="injoignable"):
        GitHubSource(GH, runner=boom).resolve_revision()


def test_github_fetch_streams_raw_file(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"a,b\n1,2\n")

    source = GitHubSource(GH, client=httpx.Client(transport=httpx.MockTransport(handler)))
    source.fetch("atp_players.csv", SHA, tmp_path / "out.csv")
    assert seen == [f"https://raw.githubusercontent.com/org/mirror/{SHA}/atp/atp_players.csv"]
    assert (tmp_path / "out.csv").read_bytes() == b"a,b\n1,2\n"


def test_github_http_error_becomes_unavailable(tmp_path: Path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    with pytest.raises(SourceUnavailableError, match="HTTPStatusError"):
        GitHubSource(GH, client=client).fetch("atp_players.csv", SHA, tmp_path / "out.csv")


def test_build_source_by_kind() -> None:
    assert isinstance(build_source(HF), HuggingFaceSource)
    assert isinstance(build_source(GH), GitHubSource)
