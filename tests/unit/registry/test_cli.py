"""`python -m tennis_mlops.registry` : état, promotion manuelle, rollback, traçabilité, refus."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from tennis_mlops.registry.__main__ import EXIT_OK, EXIT_REFUSED, main
from tennis_mlops.registry.promotion import Promoter
from tests.support.registry import MODEL, RegistryWorld, registry_world, verdict


@pytest.fixture
def world(tmp_path: Path) -> Iterator[RegistryWorld]:
    with registry_world(tmp_path) as w:
        yield w


def run(world: RegistryWorld, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--tracking-uri", world.uri, "--model", MODEL, *args])
    out, err = capsys.readouterr()
    return code, out, err


def test_status_promote_rollback_trace(world: RegistryWorld, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(world, capsys, "status")[1].startswith(f"{MODEL} : aucune version enregistrée")

    v1, v2 = world.add_version(0.6), world.add_version(0.3)
    Promoter(world.registry).apply(v1, verdict(True))
    Promoter(world.registry).apply(v2, verdict(False, compared_to=v1, delta=0.05))
    code, out, _ = run(world, capsys, "status")
    assert code == EXIT_OK
    assert f"{MODEL} — production : v{v1} · staging : v{v2}" in out
    assert "rejected" in out and "failed" in out and f"production : v{v1} — " in out

    code, out, _ = run(world, capsys, "promote", "--version", v2, "--reason", "bon en shadow")
    assert code == EXIT_OK and f"v{v2} en production (avant : v{v1}) — bon en shadow" in out

    code, out, _ = run(world, capsys, "trace")
    assert code == EXIT_OK
    assert out.startswith(f"{MODEL} v{v2} — alias : production · lifecycle : production")
    for step in ("0. prédiction", "1. version", "promotion_reason : bon en shadow", "2. run MLflow", "3. campagne",
                 "4. run Prefect", "5. données", "6. source"):  # fmt: skip
        assert step in out
    assert "dataset_version" in out

    code, out, _ = run(world, capsys, "rollback", "--reason", "plaintes")
    assert code == EXIT_OK and f"production v{v2} → v{v1} (plaintes)" in out
    assert "sans redéploiement" in out
    assert [r["event"] for r in world.journal_runs()] == ["promotion manuelle", "rollback"]

    code, out, _ = run(world, capsys, "trace", "--version", v2)
    assert "lifecycle : rolled_back" in out and "rollback_reason : plaintes" in out


def test_refusals_exit_with_code_1(world: RegistryWorld, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(world, capsys, "rollback", "--reason", "panne")
    assert code == EXIT_REFUSED and "refusé : aucune version de atp-registre en production" in err
    assert run(world, capsys, "trace")[0] == EXIT_REFUSED
    assert run(world, capsys, "promote", "--version", "7", "--reason", "x")[0] == EXIT_REFUSED
    code = main(["--tracking-uri", "http://127.0.0.1:9", "--model", MODEL, "status"])
    assert code == EXIT_REFUSED and "serveur MLflow indisponible" in capsys.readouterr().err
