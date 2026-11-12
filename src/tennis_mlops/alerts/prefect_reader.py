"""Lecture d'un flow run dans l'API Prefect : nom, état, message, première task en échec, origine du lancement.

L'automation ne transmet que trois identifiants (événement, nom de l'événement, flow run) : le service d'alertes
relit le reste ici, au moment de construire la carte.
"""

from __future__ import annotations

import re
from typing import Protocol
from uuid import UUID

from prefect import get_client
from prefect.client.schemas.filters import (
    FlowRunFilter,
    FlowRunFilterId,
    TaskRunFilter,
    TaskRunFilterState,
    TaskRunFilterStateType,
)
from prefect.client.schemas.objects import StateType
from prefect.client.schemas.sorting import TaskRunSort

from tennis_mlops.alerts.builders import FlowRunInfo

FLOW_RUN_PREFIX = "prefect.flow-run."
EXCEPTION_PREFIX = re.compile(r"^(Flow run encountered an exception|Task run encountered an exception)\s*:\s*")


class FlowRunReader(Protocol):
    """Ce dont le service d'alertes a besoin (Prefect en vrai, un dictionnaire dans les tests)."""

    def flow_run(self, flow_run_id: str) -> FlowRunInfo | None:
        """Le flow run, ou None s'il est introuvable."""
        ...


def flow_run_id_of(resource_id: str) -> str:
    """`prefect.flow-run.<uuid>` → `<uuid>` (l'automation transmet l'identifiant de la ressource)."""
    return resource_id.removeprefix(FLOW_RUN_PREFIX).strip()


def clean_message(message: str | None) -> str:
    """Retire l'en-tête générique de Prefect : reste « ValueError : … »."""
    text = EXCEPTION_PREFIX.sub("", (message or "").strip())
    name, sep, rest = text.partition(": ")
    if sep and name.isidentifier() and name[0].isupper():
        return f"{name} : {rest}"
    return text


def task_label(name: str, dynamic_key: str | None) -> str:
    """Nom lisible d'une task run : « TemporalSplitter.split-0 » → « TemporalSplitter.split ».

    Prefect suffixe le nom de la task par sa clé dynamique, sauf si la task a un `task_run_name` à elle (« contrôles
    2016-520 ») : on ne retire que ce suffixe-là.
    """
    suffix = f"-{(dynamic_key or '')[:3]}"
    return name.removesuffix(suffix) if dynamic_key and name.endswith(suffix) else name


class PrefectFlowRunReader:
    """Lit l'API Prefect (`PREFECT_API_URL`) avec le client synchrone."""

    def flow_run(self, flow_run_id: str) -> FlowRunInfo | None:
        """Flow run, flow, déploiement et première task en échec ; None si le run n'existe pas ou si l'API se tait."""
        try:
            run_uuid = UUID(flow_run_id_of(flow_run_id))
        except ValueError:
            return None
        with get_client(sync_client=True) as client:
            try:
                run = client.read_flow_run(run_uuid)
            except Exception:
                return None
            flow = client.read_flow(run.flow_id)
            deployment = client.read_deployment(run.deployment_id).name if run.deployment_id else None
            failed = client.read_task_runs(
                flow_run_filter=FlowRunFilter(id=FlowRunFilterId(any_=[run_uuid])),
                task_run_filter=TaskRunFilter(
                    state=TaskRunFilterState(type=TaskRunFilterStateType(any_=[StateType.FAILED, StateType.CRASHED]))
                ),
                sort=TaskRunSort.EXPECTED_START_TIME_ASC,
                limit=1,
            )
        state = run.state
        created_by = run.created_by.type if run.created_by else None
        return FlowRunInfo(
            id=str(run.id),
            name=run.name,
            flow_name=flow.name,
            state_type=state.type.value if state else "UNKNOWN",
            state_name=state.name if state and state.name else "inconnu",
            message=clean_message(state.message if state else ""),
            failed_task=task_label(failed[0].name, failed[0].dynamic_key) if failed else None,
            deployment=deployment,
            started_at=run.start_time,
            tags=tuple(run.tags or ()),
            automation=created_by == "AUTOMATION",
        )
