"""Test de fumée de l'interface sur la stack qui tourne : `make ui-smoke` (repris par `make scenarios`, scénario S9).

1. le service `ui` répond (`/_stcore/health`) ;
2. la page Match, exécutée telle quelle contre la vraie API, prédit deux joueurs sur terre battue et affiche leur
   probabilité et la version ;
3. la prédiction est dans la table `predictions`, avec `client = 'ui'` et le `request_id` affiché.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import httpx
import sqlalchemy as sa
from streamlit.testing.v1 import AppTest

from tennis_mlops.config import Settings

PAGE = Path(__file__).parent / "pages" / "match.py"


def run(ui_url: str, settings: Settings) -> list[str]:
    """Les étapes réussies ; lève `AssertionError` à la première qui échoue."""
    done = []
    health = httpx.get(f"{ui_url.rstrip('/')}/_stcore/health", timeout=10)
    assert health.status_code == 200, f"interface : {ui_url} répond {health.status_code}"
    done.append(f"interface en ligne ({ui_url})")

    app = AppTest.from_file(str(PAGE), default_timeout=60).run()  # client réel : TENNIS_API_URL, X-Client ui
    assert not app.exception and not app.error, f"page Match : {[e.value for e in app.error] or app.exception}"
    app.selectbox[2].set_value("Clay")
    app.button[0].click().run()
    assert not app.error and not app.warning, f"prédiction : {[e.value for e in (*app.error, *app.warning)]}"
    shown = [m.value for m in app.metric]
    assert len(shown) == 2 and all(v.endswith("%") for v in shown), f"probabilités absentes : {shown}"
    text = " ".join(c.value for c in app.caption) + " ".join(m.value for m in app.markdown)
    version = re.search(r"v(\d+)\*\*", text)
    request = re.search(r"request_id `([\w.-]+)`", text)
    assert version and request, "version ou request_id absents de la page"
    done.append(f"prédiction affichée : {' / '.join(shown)}, version v{version.group(1)}")

    engine = sa.create_engine(settings.sqlalchemy_url())
    query = sa.text("SELECT client, surface, model_version FROM predictions WHERE request_id = :r")
    for _ in range(20):  # les traces partent par paquets, toutes les 0,5 s
        with engine.connect() as conn:
            row = conn.execute(query, {"r": request.group(1)}).one_or_none()
        if row is not None:
            break
        time.sleep(0.5)
    assert row is not None, f"aucune ligne predictions pour request_id {request.group(1)}"
    assert tuple(row) == ("ui", "Clay", version.group(1)), f"ligne inattendue : {tuple(row)}"
    done.append(f"tracée dans predictions : client ui, request_id {request.group(1)}")
    return done


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée : code de sortie 1 si une étape échoue."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.ui.smoke", description=__doc__)
    parser.add_argument("--ui-url", default="http://127.0.0.1:8501")
    args = parser.parse_args(argv)
    try:
        for step in run(args.ui_url, Settings()):
            print(f"✅ {step}")
    except (AssertionError, httpx.HTTPError, sa.exc.SQLAlchemyError) as exc:
        print(f"❌ {type(exc).__name__} : {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
