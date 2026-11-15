"""Page Match : deux joueurs, une surface, un tournoi → la probabilité de chacun, la version, et pourquoi."""

from __future__ import annotations

import streamlit as st

from tennis_mlops.serving.schemas import MatchRequest, PlayerInput, PredictResponse
from tennis_mlops.ui import components, views

st.title("🎾 Prédire un match")
st.caption("Un match à venir, prédit par la version en production. Rien n'est inventé : les profils et l'historique "
           "viennent des matchs validés en base.")  # fmt: skip

left, right = st.columns(2)
with left:
    player1 = components.player_picker("Joueur 1", "p1", default_rank=1)
with right:
    player2 = components.player_picker("Joueur 2", "p2", default_rank=2)

c1, c2, c3, c4 = st.columns(4)
surface = c1.selectbox("Surface", ["Clay", "Hard", "Grass", "Carpet"], format_func=views.surface_label)
level = c2.selectbox("Tournoi", ["G", "M", "A", "F", "D", "O"], index=1, format_func=views.level_label)
round_ = c3.selectbox("Tour", ["F", "SF", "QF", "R16", "R32", "R64", "R128", "RR"], index=2,
                      format_func=views.round_label)  # fmt: skip
best_of = c4.selectbox("Format", [3, 5], index=[3, 5].index(views.default_best_of(level)),
                       format_func=lambda n: f"{n} sets gagnants" if n == 5 else "2 sets gagnants")  # fmt: skip

current = (getattr(player1, "id", None), getattr(player2, "id", None), surface, level, round_, best_of)
same = player1 is not None and player2 is not None and player1.id == player2.id
if same:
    st.warning("Choisir deux joueurs différents.")

if st.button("Prédire", type="primary", disabled=player1 is None or player2 is None or same):
    assert player1 is not None and player2 is not None
    request = MatchRequest(surface=surface, tourney_level=level, round=round_, best_of=best_of,  # type: ignore[arg-type]
                           player1=PlayerInput(id=player1.id), player2=PlayerInput(id=player2.id))  # fmt: skip
    made, elapsed = components.timed(lambda api: api.predict(request))
    if made is not None:
        first, second = player1.id, player2.id
        stats = (components.call(lambda api: api.player_stats(first), quiet=True),
                 components.call(lambda api: api.player_stats(second), quiet=True))  # fmt: skip
        h2h = components.call(lambda api: api.h2h(first, second), quiet=True)
        st.session_state["match_result"] = (made, elapsed, stats, h2h, surface, current)

result = st.session_state.get("match_result")
if result is not None and result[5] == current:  # un autre choix efface l'ancienne prédiction
    prediction: PredictResponse = result[0]
    round_trip, stats, h2h, shown_surface = result[1], result[2], result[3], result[4]
    st.divider()
    a, b = st.columns(2)
    for column, player in ((a, prediction.player1), (b, prediction.player2)):
        favourite = player.id == prediction.predicted_winner_id
        column.metric(f"{'⭐ ' if favourite else ''}{player.name or player.id}", views.percent(player.win_probability))
        column.progress(player.win_probability)
    st.markdown(
        f"Modèle **{prediction.model.name} v{prediction.model.version}** (alias `{prediction.model.alias}`) · "
        f"match du **{prediction.match_date}**, historique jusqu'au {prediction.features_as_of} · "
        f"{views.number(prediction.latency_ms, 1)} ms dans l'API, {views.number(round_trip, 0)} ms aller-retour"
    )
    st.caption(f"request_id `{prediction.request_id}` : retrouvable dans la table `predictions` (client `ui`).")
    for warning in prediction.warnings:
        st.info(warning)
    st.subheader("Ce que voit le modèle")
    st.dataframe(views.comparison(prediction, stats, h2h, shown_surface), width="stretch")
    st.caption("Écart : joueur 1 moins joueur 2, tel qu'il entre dans le modèle (le classement en log du rapport).")
    with st.expander("Toutes les features envoyées au modèle"):
        st.json(prediction.features)
