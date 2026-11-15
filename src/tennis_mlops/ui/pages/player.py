"""Page Joueur : classement dans le temps, bilans par surface et par niveau, forme, derniers résultats."""

from __future__ import annotations

from datetime import date

import altair as alt
import streamlit as st

from tennis_mlops.ui import components, views

st.title("👤 Joueur")
player = components.player_picker("Joueur", "player")
if player is None:
    st.stop()

limits = components.call(lambda api: api.players(limit=1), quiet=True)
last_day = limits.as_of if limits else None
as_of = st.date_input("Statistiques au", value=last_day, max_value=last_day, format="DD/MM/YYYY",
                      help="Jamais après la dernière journée de l'historique validé.")  # fmt: skip
day = as_of if isinstance(as_of, date) else None
chosen = player.id
stats = components.call(lambda api: api.player_stats(chosen, day))
if stats is None:
    st.stop()

info = stats.player
st.subheader(views.player_label(info))
m = st.columns(5)
m[0].metric("Classement", views.number(info.rank))
m[1].metric("Points", views.number(info.rank_points))
m[2].metric("Âge", views.number(info.age, 1))
m[3].metric("Elo général", views.number(stats.ratings.elo) if stats.ratings else "—",
            help="Ce que verrait le modèle pour un match du lendemain (à la dernière journée seulement).")  # fmt: skip
m[4].metric("Matchs joués", info.matches)
st.caption(f"Au {stats.as_of} · historique jusqu'au {stats.history_last_day} · main {info.hand or '?'} · "
           f"{views.number(info.height_cm)} cm · dernier match le {info.last_match or '—'}")  # fmt: skip

rankings = views.rankings_frame(stats)
if not rankings.empty:
    st.markdown("#### Classement au début de chaque tournoi")
    chart = (
        alt.Chart(rankings)
        .mark_line(point=alt.OverlayMarkDef(color=views.CLAY), interpolate="step-after", color=views.CLAY)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
            y=alt.Y(
                "classement:Q",
                scale=alt.Scale(reverse=True, zero=False),
                title="classement (1 en haut)",
                axis=alt.Axis(format="d", tickMinStep=1),
            ),
            tooltip=["date:T", "classement:Q", "points:Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")

left, right = st.columns(2)
with left:
    st.markdown("#### Bilans")
    st.dataframe(views.records_frame(stats), hide_index=True, width="stretch",
                 column_config={"% victoires": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                               max_value=1)})  # fmt: skip
with right:
    st.markdown(f"#### Forme : {stats.form.wins} victoires sur {len(stats.form.results)}")
    st.markdown(" ".join("🟩" if won else "🟥" for won in stats.form.results) or "aucun match")
    st.caption("Du plus ancien au plus récent.")
    if stats.ratings and stats.ratings.surface_elo:
        st.markdown("#### Elo par surface")
        st.dataframe({views.surface_label(k): {"Elo": v, "matchs": stats.ratings.surface_matches.get(k, 0)}
                      for k, v in stats.ratings.surface_elo.items()}, width="stretch")  # fmt: skip

st.markdown("#### Derniers résultats")
st.dataframe(views.recent_frame(stats), hide_index=True, width="stretch")
