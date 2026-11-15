"""Ce que l'interface affiche, sans Streamlit : messages d'erreur, libellés, tableaux."""

from __future__ import annotations

from pathlib import Path

import pytest

from tennis_mlops.api.client import ApiError
from tennis_mlops.ui import views
from tennis_mlops.ui.__main__ import command
from tests.support.serving import ServingWorld, match_body, serving_world


@pytest.mark.parametrize(
    ("error", "level", "hint"),
    [
        (ApiError(None, "unreachable", "GET /ready : ConnectError"), "error", "make up"),
        (ApiError(503, "unavailable", "aucun modèle servi : aucune version désignée"), "warning", "make train"),
        (ApiError(503, "unavailable", "historique des matchs pas encore chargé"), "warning", "make ingest"),
        (ApiError(503, "unavailable", "features_version 2 au lieu de 1 : réentraîner"), "warning", "make train"),
        (ApiError(503, "unavailable", "autre chose"), "warning", "quelques secondes"),
        (ApiError(404, "unknown_player", "joueur 9 : aucun match"), "warning", "débutant"),
    ],
)
def test_errors_are_explained(error: ApiError, level: str, hint: str) -> None:
    notice = views.explain(error, "http://api")
    assert notice.level == level and hint in (notice.hint or "")


def test_other_errors() -> None:
    assert views.explain(ApiError(422, "invalid_match", "match du 2015"), "u").title == "Requête refusée par l'API"
    assert views.explain(ApiError(500, "internal", "boom"), "u").level == "error"


def test_labels_and_numbers() -> None:
    assert views.surface_label("Clay") == "Terre battue" and views.surface_label(None) == "surface inconnue"
    assert views.level_label("G") == "Grand Chelem" and views.round_label("QF") == "Quart de finale"
    assert views.default_best_of("G") == 5 and views.default_best_of("M") == 3
    assert views.percent(0.6123) == "61,2 %" and views.percent(None) == "—" and views.percent(float("nan")) == "—"
    assert views.number(12345.6) == "12 346" and views.number(None) == "—" and views.number(1.25, 1) == "1,2"
    assert views.form_string([True, False]) == "V D" and views.form_string([]) == "aucun match"


def test_tables(tmp_path: Path) -> None:
    world: ServingWorld = serving_world(tmp_path).ready()
    stats = world.service.player_stats(100_001)
    assert list(views.records_frame(stats)["périmètre"].unique()) == ["Général", "Surface", "Niveau"]
    assert len(views.recent_frame(stats)) == 15 and not views.rankings_frame(stats).empty
    h2h = world.service.head_to_head(100_001, 100_002)
    assert len(views.meetings_frame(h2h)) == h2h.total and len(views.h2h_surfaces_frame(h2h)) == len(h2h.by_surface)
    from tennis_mlops.serving.schemas import PredictResponse

    prediction = PredictResponse.model_validate(world.client.post("/predict", json=match_body()).json())
    table = views.comparison(prediction, (stats, None), h2h, "Clay")
    assert table.loc["Elo terre battue", "Joueur 2"] == "1 500 (départ)"  # pas de statistiques : Elo de départ
    assert table.loc["Face-à-face", "Joueur 1"] == str(h2h.p1_wins)
    world.client.post("/predict/batch", json={"matches": [match_body(match_id="2099-1-1")]},
                      headers={"X-Client": "scoring"})  # fmt: skip
    world.flush()
    upcoming, played = views.fixtures_frames(world.service.fixture_predictions())
    assert len(upcoming) == 1 and played.empty and upcoming.iloc[0]["état"] == "à jouer"


def test_streamlit_command() -> None:
    args = command("0.0.0.0", 8501)
    assert args[:2] == ["streamlit", "run"] and args[2].endswith("ui/app.py")
    assert "--browser.gatherUsageStats" in args and args[args.index("--server.port") + 1] == "8501"


def test_smoke_against_an_in_memory_stack(page: object, world: ServingWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    """`make ui-smoke` : la page Match, une prédiction, la ligne `predictions` retrouvée par son request_id."""
    from types import SimpleNamespace

    from tennis_mlops.config import Settings
    from tennis_mlops.ui import smoke

    monkeypatch.setattr("tennis_mlops.ui.smoke.httpx.get", lambda *_, **__: SimpleNamespace(status_code=200))
    settings = Settings(postgres_dsn=f"sqlite:///{world.path / 'tennis.sqlite'}")  # type: ignore[arg-type]
    world.service.writer.start()  # les traces partent en arrière-plan, comme dans la stack
    try:
        steps = smoke.run("http://ui", settings)
    finally:
        world.service.writer.stop()
    assert len(steps) == 3 and "client ui" in steps[-1]
    monkeypatch.setattr("tennis_mlops.ui.smoke.httpx.get", lambda *_, **__: SimpleNamespace(status_code=502))
    with pytest.raises(AssertionError, match="502"):
        smoke.run("http://ui", settings)
