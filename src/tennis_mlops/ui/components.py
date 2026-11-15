"""Briques Streamlit partagées par les pages : client de l'API, appels avec erreurs lisibles, choix d'un joueur.

L'interface ne parle qu'à l'API (jamais à la base ni à MLflow), en se présentant avec `X-Client: ui` : ses
prédictions sont tracées et comptées comme celles de n'importe quel client.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import streamlit as st

from tennis_mlops.api.client import ApiClient, ApiError
from tennis_mlops.config import Settings
from tennis_mlops.serving.schemas import PlayerSummary
from tennis_mlops.ui import views

CLIENT_NAME = "ui"


def make_client() -> ApiClient:
    """Le client de l'interface (remplacé dans les tests par un client branché sur une API en mémoire)."""
    return ApiClient(str(Settings().api_url), CLIENT_NAME, timeout_s=15)


def client() -> ApiClient:
    """Un client par session de navigateur."""
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = make_client()
    api: ApiClient = st.session_state["api_client"]
    return api


def call[T](fn: Callable[[ApiClient], T], quiet: bool = False) -> T | None:
    """Appelle l'API ; en cas d'erreur, l'explique à l'écran (sauf `quiet`) et renvoie None."""
    api = client()
    try:
        return fn(api)
    except ApiError as exc:
        if not quiet:
            show(views.explain(exc, api.base_url))
        return None


def timed[T](fn: Callable[[ApiClient], T]) -> tuple[T | None, float]:
    """Comme `call`, avec la durée de l'aller-retour vu du navigateur (ms)."""
    started = time.perf_counter()
    result = call(fn)
    return result, (time.perf_counter() - started) * 1000


def show(notice: views.Notice) -> None:
    """Affiche un message d'erreur ou d'avertissement."""
    text = f"**{notice.title}** — {notice.text}"
    if notice.hint:
        text += f"\n\n{notice.hint}"
    {"error": st.error, "warning": st.warning, "info": st.info}[notice.level](text)


def player_picker(label: str, key: str, default_rank: int = 1) -> PlayerSummary | None:
    """Un joueur : recherche par nom ou identifiant ; sans texte, les joueurs classés actifs.

    Un identifiant absent de l'historique (débutant) est proposé quand même : l'API le prédit avec l'Elo de départ.
    """
    search = st.text_input(label, key=f"{key}_search", placeholder="Nom (ex. « nadal ») ou identifiant Sackmann")
    found = call(lambda api: api.players(search.strip() or None, limit=50))
    if found is None:
        return None
    options = list(found.players)
    if not options:
        text = search.strip()
        if text.isdigit():
            st.caption(
                f"Identifiant {text} : aucun match dans l'historique (débutant ?) ; prédit avec l'Elo de départ."
            )
            return PlayerSummary(id=int(text), name=f"joueur {text}")
        st.warning(f"Aucun joueur ne correspond à « {text} » dans l'historique validé.")
        return None
    by_id = {p.id: p for p in options}
    index = min(default_rank - 1, len(options) - 1) if not search.strip() else 0
    choice = st.selectbox(
        "Joueur", list(by_id), index=index, key=f"{key}_choice", format_func=lambda pid: views.player_label(by_id[pid]),
        label_visibility="collapsed",
    )  # fmt: skip
    return by_id.get(choice) if choice is not None else None


def status_sidebar() -> None:
    """État de l'API dans la barre latérale : prête ou non, version servie, horloge simulée."""
    ready = call(lambda api: api.ready(), quiet=True)
    if ready is None:
        st.sidebar.error("API injoignable")
        return
    checks = ready.get("checks", {})
    st.sidebar.markdown(f"**API** : {views.READINESS.get(ready.get('status', ''), ready.get('status'))}")
    version = checks.get("model")
    st.sidebar.markdown(f"**Modèle** : {'v' + str(version) if version else 'aucun en production'}")
    st.sidebar.markdown(f"**Historique jusqu'au** : {checks.get('features_last_day') or '—'}")
    for reason in ready.get("reasons", []):
        st.sidebar.caption(reason)
