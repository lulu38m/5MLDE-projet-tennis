"""Page Modèle : la version en production, sa qualité hors ligne et en ligne, l'horloge simulée, les liens."""

from __future__ import annotations

import streamlit as st

from tennis_mlops.config import Settings
from tennis_mlops.ui import components, views

st.title("🧠 Modèle en production")
ready = components.call(lambda api: api.ready())
info = components.call(lambda api: api.model())
if ready is None or info is None:
    st.stop()

st.markdown(f"**État de l'API** : {views.READINESS.get(ready['status'], ready['status'])}")
for reason in ready.get("reasons", []):
    st.warning(reason)

served = info.get("served")
features = info.get("features") or {}
online = info.get("online") or {}
if served:
    tags = served.get("tags", {})
    m = st.columns(4)
    m[0].metric("Version", f"v{served['version']}")
    candidate = str(tags.get("candidate") or "—")
    m[1].metric("Famille", candidate.split()[0], help=f"Candidat : {candidate}")
    loss = tags.get("test.log_loss")
    m[2].metric("Log loss (test)", views.number(float(loss), 3) if loss else "—")
    m[3].metric("Mise en service", (tags.get("promoted_at") or "—")[:10], help=tags.get("decided_by"))
    st.caption(f"{served['name']} · alias `{served['alias']}` · run `{served['run_id']}` · données jusqu'au "
               f"{tags.get('data_window_end', '—')} · features v{tags.get('features_version', '?')}")  # fmt: skip
else:
    st.warning("Aucune version désignée par l'alias `production` : `make train`.")

st.markdown("#### Qualité en ligne (matchs dont le résultat est connu)")
o = st.columns(4)
o[0].metric("Prédictions", online.get("predictions", 0))
o[1].metric("Avec résultat", online.get("with_feedback", 0), help=f"couverture {views.percent(online.get('coverage'))}")
o[2].metric("Accuracy", views.percent(online.get("accuracy")))
o[3].metric("Log loss", views.number(online.get("log_loss"), 3))

st.markdown("#### Horloge et historique")
h = st.columns(4)
h[0].metric("Horloge simulée", features.get("watermark") or "—", help="Jusqu'où la dernière ingestion est allée")
h[1].metric("Dernier match connu", features.get("last_day") or "—")
h[2].metric("Matchs", views.number(features.get("matches")))
h[3].metric("Joueurs", views.number(features.get("players")))

settings = Settings()
registry = info.get("registry") or {}
st.markdown("#### Liens")
st.markdown(
    f"- [MLflow : le registre `atp-winner`]({settings.mlflow_ui_url}#/models/atp-winner)\n"
    f"- [Prefect : les runs de la simulation]({settings.prefect_ui_url}runs)\n"
    f"- [Grafana : les dashboards]({settings.grafana_ui_url}) (phase 10)\n"
    f"- [API : documentation interactive]({settings.api_ui_url}docs)"
)
st.caption(f"Registre relu le {registry.get('last_success') or '—'} · {registry.get('reloads', 0)} rechargement(s)")
