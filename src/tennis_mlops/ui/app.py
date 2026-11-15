"""Interface web de test : `python -m tennis_mlops.ui` (service `ui` de la stack, http://127.0.0.1:8501).

Elle ne parle qu'à l'API de prédiction (`X-Client: ui`) : aucune base, aucun MLflow. Cinq pages, choisies dans la
barre latérale ; l'état de l'API (prête, version servie, historique) y reste affiché.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from tennis_mlops.ui import components

HERE = Path(__file__).parent / "pages"

st.set_page_config(page_title="tennis-mlops", page_icon="🎾", layout="wide")
navigation = st.navigation([
    st.Page(HERE / "match.py", title="Match", icon="🎾", default=True),
    st.Page(HERE / "player.py", title="Joueur", icon="👤"),
    st.Page(HERE / "h2h.py", title="Face-à-face", icon="🤝"),
    st.Page(HERE / "fixtures.py", title="Matchs à venir", icon="📅"),
    st.Page(HERE / "model.py", title="Modèle", icon="🧠"),
])  # fmt: skip
components.status_sidebar()
navigation.run()
