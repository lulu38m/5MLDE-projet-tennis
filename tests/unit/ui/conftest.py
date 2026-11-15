"""L'interface branchée sur une API en mémoire : vraie application FastAPI, vrai service, base SQLite."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tennis_mlops.api.client import ApiClient
from tennis_mlops.ui import components
from tests.support.serving import ServingWorld, serving_world

PAGES = Path(components.__file__).parent / "pages"
APP = PAGES.parent / "app.py"


@pytest.fixture
def world(tmp_path: Path) -> ServingWorld:
    return serving_world(tmp_path).ready()


@pytest.fixture
def page(world: ServingWorld, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], AppTest]]:
    """`page("match")` : la page, dont le client parle à l'API en mémoire (avec `X-Client: ui`)."""
    monkeypatch.setattr(components, "make_client", lambda: ApiClient("http://api", "ui", http=world.client))

    def run(name: str) -> AppTest:
        app = AppTest.from_file(str(APP if name == "app" else PAGES / f"{name}.py"), default_timeout=60)
        return app.run()

    yield run
