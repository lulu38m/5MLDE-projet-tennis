"""Page Face-à-face : les rencontres de deux joueurs et leur bilan par surface."""

from __future__ import annotations

import streamlit as st

from tennis_mlops.ui import components, views

st.title("🤝 Face-à-face")
left, right = st.columns(2)
with left:
    player1 = components.player_picker("Joueur 1", "h1", default_rank=1)
with right:
    player2 = components.player_picker("Joueur 2", "h2", default_rank=2)
if player1 is None or player2 is None:
    st.stop()
if player1.id == player2.id:
    st.warning("Choisir deux joueurs différents.")
    st.stop()

first, second = player1.id, player2.id
h2h = components.call(lambda api: api.h2h(first, second))
if h2h is None:
    st.stop()

name1, name2 = h2h.player1.name or str(player1.id), h2h.player2.name or str(player2.id)
a, b, c = st.columns(3)
a.metric(name1, h2h.p1_wins)
b.metric("Rencontres", h2h.total)
c.metric(name2, h2h.p2_wins)
st.caption(f"Matchs validés jusqu'au {h2h.as_of} (forfaits exclus, comme pour le modèle).")
if h2h.total == 0:
    st.info("Ces deux joueurs ne se sont jamais rencontrés dans l'historique.")
    st.stop()

st.markdown("#### Par surface")
st.bar_chart(
    views.h2h_surfaces_frame(h2h).set_index("surface"),
    horizontal=True,
    stack=True,
    height=180,
    color=[views.CLAY, "#5f7f6e"],
)
st.markdown("#### Les rencontres")
st.dataframe(views.meetings_frame(h2h), hide_index=True, width="stretch")
