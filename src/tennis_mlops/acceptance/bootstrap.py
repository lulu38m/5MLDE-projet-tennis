"""Fin de `make up` : la stack démarrée sert un premier modèle, sans autre commande.

Sur une stack neuve, la chaîne fait elle-même le travail, comme pendant la démo : une ingestion amorce l'historique
2010 → 2015 (contrôlé par Great Expectations), son volume déclenche l'événement `tennis.training.requested`,
l'automation Prefect lance la campagne sur le worker, et la règle de mise en service place le meilleur candidat en
`production`. On attend que l'API le serve.

Idempotent : si une version est déjà en production, on vérifie seulement que l'API la sert.
"""

from __future__ import annotations

from collections.abc import Callable

from tennis_mlops.acceptance.scenarios import Actions, settle
from tennis_mlops.acceptance.stack import Stack, now, wait_for
from tennis_mlops.registry.tagging import Alias

TRAINING_DEPLOYMENT = "entrainement-atp/sur-evenement"
API_TIMEOUT_S = 180
EXIT_OK, EXIT_FAILED = 0, 1


class BootstrapError(RuntimeError):
    """Pas de premier modèle (ingestion refusée, entraînement en échec, version refusée par la règle)."""


def bootstrap(stack: Stack, actions: Actions, log: Callable[[str], None] = print,
              request_training: Callable[[], None] | None = None) -> str:  # fmt: skip
    """Renvoie la version servie ; `BootstrapError` si la chaîne n'a pas pu en produire une."""
    production = stack.alias(Alias.PRODUCTION)
    if production is None:
        start = now()
        start_day = stack.config.simulation.replay_start
        if stack.watermark() < start_day:
            log(f"premier passage d'ingestion : historique jusqu'au {start_day}, contrôlé par Great Expectations…")
            outcome = actions.ingest()
            summary = getattr(outcome.result, "summary", lambda: outcome.state_name)()
            if outcome.failed or outcome.state_name != "Completed":
                msg = f"ingestion initiale : {outcome.state_name} ({summary}) — voir Prefect, puis relancer make up"
                raise BootstrapError(msg)
            log(f"  {summary}")
        if not stack.events("training_requested", start):
            log("historique déjà ingéré mais aucun modèle : campagne lancée sur le worker")
            (request_training or _run_training_deployment)()
        log("entraînement lancé par l'automation Prefect sur le worker (1 à 3 min)…")
        runs = settle(stack, start)
        log("  " + ", ".join(f"{r.name} : {r.state_name}" for r in runs))
        production = stack.alias(Alias.PRODUCTION)
        if production is None:
            latest = stack.latest_version()
            reason = stack.version_tags(latest).get("gate_reason", "") if latest else "aucune version enregistrée"
            msg = f"aucune version mise en production : {reason} — voir {stack.url('prefect')} et make registry"
            raise BootstrapError(msg)
    wait_for(lambda: stack.served_version() == production, API_TIMEOUT_S, f"API sur v{production}", interval_s=3)
    tags = stack.version_tags(production)
    log(f"modèle en production : v{production} (log loss test {tags.get('test.log_loss', '?')}, données jusqu'au "
        f"{tags.get('data_window_end', '?')}), servi par l'API")  # fmt: skip
    return production


def _run_training_deployment() -> None:
    from prefect.deployments import run_deployment

    run_deployment(TRAINING_DEPLOYMENT, timeout=0)  # n'attend pas : `settle` suit le run
