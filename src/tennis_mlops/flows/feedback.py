"""Flow Prefect de feedback : remonter à l'API le vrai résultat des matchs qu'elle a prédits.

```
predictions ⋈ matches (résultats validés), sauf feedbacks ─► POST /feedback (X-Client: feedback) ─► qualité en ligne
```

- **Seulement des résultats validés** : on lit la table `matches`, alimentée par les lots acceptés par Great
  Expectations, jamais la source brute. Un lot en quarantaine ne remonte rien tant qu'il n'est pas corrigé.
- **Seulement des matchs joués** : un forfait n'a pas de vainqueur sur le court, il ne juge aucune prédiction.
- **Idempotent** : un résultat déjà remonté n'est plus proposé ; le renvoyer ne ferait que le remplacer.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from pydantic import BaseModel

from tennis_mlops.api.client import ApiClient, ApiError
from tennis_mlops.config import DEFAULT_CONFIG_PATH, Settings
from tennis_mlops.serving.persistence import ApiStore

FLOW_NAME = "feedback-atp"
CLIENT_NAME = "feedback"
INVALID = 422


class FeedbackReport(BaseModel):
    """Bilan d'un passage de feedback."""

    awaiting: int = 0
    sent: int = 0
    rejected: int = 0
    judged: int = 0  # prédictions jugées (un match peut avoir été prédit par plusieurs versions)
    correct: int = 0
    by_version: dict[str, int] = {}

    def summary(self) -> str:
        """Une ligne pour les logs."""
        if not self.awaiting:
            return "aucun résultat à remonter"
        rate = f", {self.correct / self.judged:.0%} justes" if self.judged else ""
        rejected = f", {self.rejected} refusé(s)" if self.rejected else ""
        return f"{self.sent} résultat(s) remonté(s) ; {self.judged} prédiction(s) jugée(s){rate}{rejected}"


class FeedbackDependencies:
    """API et base ; les tests fournissent les leurs."""

    def __init__(self, api: ApiClient, store: ApiStore) -> None:
        self.api, self.store = api, store


def default_dependencies(settings: Settings) -> FeedbackDependencies:
    """Les services de la stack."""
    return FeedbackDependencies(
        ApiClient(str(settings.api_url), CLIENT_NAME), ApiStore.from_url(settings.sqlalchemy_url())
    )


dependency_factory: Callable[[Settings], FeedbackDependencies] = default_dependencies


@contextmanager
def _dependencies() -> Iterator[FeedbackDependencies]:
    deps = dependency_factory(Settings())
    try:
        yield deps
    finally:
        deps.api.close()


@task(name="résultats à remonter", cache_policy=NO_CACHE)
def awaiting(store: ApiStore, limit: int) -> list[tuple[str, int]]:
    """Matchs prédits dont le résultat validé est en base et pas encore remonté."""
    return store.results_awaiting_feedback(limit)


@task(name="POST /feedback", task_run_name="remonter {count} résultat(s)", cache_policy=NO_CACHE)
def send(api: ApiClient, results: list[tuple[str, int]], count: int) -> FeedbackReport:
    """Un appel par match ; un résultat incohérent (422) est compté et n'arrête pas les autres."""
    report = FeedbackReport(awaiting=count)
    versions: Counter[str] = Counter()
    for match_id, winner_id in results:
        try:
            response = api.feedback(match_id, winner_id)
        except ApiError as exc:
            if exc.status != INVALID:
                raise
            report.rejected += 1
            get_run_logger().warning("résultat refusé (%s) : %s", match_id, exc.message)
            continue
        report.sent += 1
        for outcome in response.predictions:
            report.judged += 1
            report.correct += int(outcome.correct)
            versions[outcome.model_version] += 1
    report.by_version = dict(versions)
    return report


@flow(name=FLOW_NAME)
def feedback_flow(limit: int = 1000) -> FeedbackReport:
    """Remonte les vrais résultats des matchs prédits."""
    logger = get_run_logger()
    with _dependencies() as deps:
        results = awaiting(deps.store, limit)
        report = send(deps.api, results, len(results)) if results else FeedbackReport()
    logger.info(report.summary())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m tennis_mlops.flows.feedback` : un passage, sans planning (`make feedback`)."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.flows.feedback", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    print(feedback_flow(args.limit).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
