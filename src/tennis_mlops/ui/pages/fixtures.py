"""Page Matchs à venir : ce que le scoring a prédit à l'avance, et une fois joué, si le modèle avait raison."""

from __future__ import annotations

import streamlit as st

from tennis_mlops.ui import components, views

st.title("📅 Matchs à venir")
st.caption("Chaque pas de `make demo` prédit les matchs de la semaine suivante (flow `scoring-atp`) ; quand leur "
           "tournoi est ingéré, le vrai résultat apparaît ici, comme dans le feedback.")  # fmt: skip
days = st.slider("Matchs datés depuis (jours avant l'horloge simulée)", 1, 60, 14)
response = components.call(lambda api: api.fixture_predictions(days=days))
if response is None:
    st.stop()

m = st.columns(4)
m[0].metric("Matchs prédits", response.count)
m[1].metric("Joués", response.played)
m[2].metric("Le modèle avait raison", response.correct)
m[3].metric("Accuracy", views.percent(response.accuracy))
st.caption(f"Horloge simulée (dernière ingestion) : {response.watermark or '—'} · historique jusqu'au "
           f"{response.as_of or '—'} · depuis le {response.since}")  # fmt: skip

if response.count == 0:
    st.info("Aucun match prédit à l'avance sur cette période : lancer `make demo`, ou `make score` pour un passage.")
    st.stop()

upcoming, played = views.fixtures_frames(response)
probability = st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
st.markdown(f"#### À jouer ({len(upcoming)})")
st.dataframe(upcoming, hide_index=True, width="stretch",
             column_config={"proba 1": probability})  # fmt: skip
st.caption("proba 1 : probabilité que le joueur 1 gagne (celle du joueur 2 est son complément).")
st.markdown(f"#### Joués ({len(played)})")
st.dataframe(played, hide_index=True, width="stretch", column_config={"proba du vainqueur": probability})
