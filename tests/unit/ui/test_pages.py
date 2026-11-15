"""Les pages de l'interface, rejouées par `AppTest` contre l'API en mémoire (aucun serveur, aucun navigateur)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tennis_mlops.api.client import ApiClient
from tennis_mlops.ui import components
from tests.support.serving import ServingWorld, match_body, serving_world
from tests.unit.ui.conftest import APP, PAGES

Page = Callable[[str], AppTest]


def _metrics(app: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in app.metric}


def test_s9_a_prediction_made_from_the_ui_is_traced_as_ui(page: Page, world: ServingWorld) -> None:
    """Scénario S9 : deux joueurs, terre battue → probabilité, version, et une ligne `predictions` client `ui`."""
    app = page("match")
    assert not app.exception and app.selectbox(key="p1_choice").value == 100_034  # n°1 par défaut
    app.selectbox[2].set_value("Clay")  # surface (après les deux joueurs)
    app.button[0].click().run()
    assert not app.exception and not app.error
    probabilities = list(_metrics(app).values())
    assert len(probabilities) == 2 and all(v.endswith("%") for v in probabilities)
    assert any("atp-winner v1" in md.value for md in app.markdown)
    assert any("Écart vu par le modèle" in str(df.value.columns.tolist()) for df in app.dataframe)
    world.flush()
    (row,) = world.rows("SELECT client, surface, p1_id, p2_id, model_version FROM predictions")
    assert row == {"client": "ui", "surface": "Clay", "p1_id": 100_034, "p2_id": 100_007, "model_version": "1"}
    clients = {r["client"] for r in world.rows("SELECT client FROM api_requests")}
    assert clients == {"ui"}  # même les recherches de joueurs se présentent comme l'interface


def test_search_a_player_by_name_and_change_the_selection(page: Page) -> None:
    app = page("match")
    app.text_input(key="p1_search").set_value("joueur 12").run()
    assert app.selectbox(key="p1_choice").value == 100_012
    app.text_input(key="p2_search").set_value("personne").run()
    assert any("Aucun joueur" in w.value for w in app.warning)
    assert app.button[0].disabled


def test_a_rookie_can_still_be_predicted(page: Page, world: ServingWorld) -> None:
    app = page("match")
    app.text_input(key="p2_search").set_value("999999").run()
    assert any("débutant" in c.value for c in app.caption)
    app.button[0].click().run()
    assert any("Elo de départ" in i.value for i in app.info)  # l'avertissement de l'API, affiché tel quel
    world.flush()
    assert world.rows("SELECT p2_id FROM predictions") == [{"p2_id": 999_999}]


def test_same_player_twice_is_refused_before_calling_the_api(page: Page) -> None:
    app = page("match")
    app.text_input(key="p2_search").set_value("100034").run()
    assert any("différents" in w.value for w in app.warning) and app.button[0].disabled


def test_player_page(page: Page, world: ServingWorld) -> None:
    app = page("player")
    assert not app.exception
    metrics = _metrics(app)
    assert metrics["Classement"] == "1" and metrics["Matchs joués"] == "100"
    assert len(app.dataframe) >= 2  # bilans et derniers résultats
    app.date_input[0].set_value(world.last_day - timedelta(days=300)).run()
    assert int(_metrics(app)["Matchs joués"]) < 100 and _metrics(app)["Elo général"] == "—"


def test_h2h_page(page: Page) -> None:
    app = page("h2h")
    assert not app.exception
    metrics = _metrics(app)
    assert int(metrics["Rencontres"]) == int(metrics["Joueur 34"]) + int(metrics["Joueur 7"])


def test_fixtures_page(page: Page, world: ServingWorld) -> None:
    app = page("fixtures")
    assert any("Aucun match prédit" in i.value for i in app.info)
    world.client.post("/predict/batch", json={"matches": [match_body(match_id="2099-5-1", tourney_name="Open")]},
                      headers={"X-Client": "scoring"})  # fmt: skip
    world.flush()
    app = page("fixtures")
    assert _metrics(app)["Matchs prédits"] == "1" and not app.exception


def test_model_page(page: Page) -> None:
    app = page("model")
    assert not app.exception
    assert _metrics(app)["Version"] == "v1"
    assert any("MLflow" in md.value and "Prefect" in md.value for md in app.markdown)


def test_the_whole_app_with_its_sidebar(page: Page) -> None:
    app = page("app")
    assert not app.exception and app.title[0].value.startswith("🎾")
    assert any("prêt" in md.value for md in app.sidebar.markdown)


@pytest.mark.parametrize("name", ["match", "player", "h2h", "fixtures", "model"])
def test_api_unreachable_is_said_plainly(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(components, "make_client", lambda: ApiClient("http://127.0.0.1:9", "ui", timeout_s=1))
    app = AppTest.from_file(str(PAGES / f"{name}.py"), default_timeout=30).run()
    assert not app.exception
    assert any("API injoignable" in e.value and "make up" in e.value for e in app.error)


def test_no_model_in_production(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = serving_world(tmp_path)
    world.registry.production = None
    world.ready()
    monkeypatch.setattr(components, "make_client", lambda: ApiClient("http://api", "ui", http=world.client))
    app = AppTest.from_file(str(PAGES / "match.py"), default_timeout=30).run()
    app.button[0].click().run()
    assert any("make train" in w.value for w in app.warning)
    sidebar = AppTest.from_file(str(APP), default_timeout=30).run()
    assert any("pas prêt" in md.value for md in sidebar.sidebar.markdown)
    player = AppTest.from_file(str(PAGES / "player.py"), default_timeout=30).run()
    assert _metrics(player)["Classement"] == "1"  # l'exploration marche sans modèle
