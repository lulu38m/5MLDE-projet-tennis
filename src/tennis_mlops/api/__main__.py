"""`python -m tennis_mlops.api serve|outcomes` : lancer l'API, ou lire ce qu'elle a tracé.

- `serve [--host 0.0.0.0 --port 8000]` : l'API (service `api` de la stack, `make api` en local).
- `outcomes [--limit 10]` : les dernières prédictions avec leur version, la durée de leur requête et le vrai résultat
  (`make predictions`) ; c'est la vue `prediction_outcomes`.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig, Settings
from tennis_mlops.log import configure_logging

EXIT_OK, EXIT_FAILED = 0, 1
DEFAULT_PORT = 8000


def build_app(settings: Settings | None = None, config: ProjectConfig | None = None):  # type: ignore[no-untyped-def]  # noqa: ANN201
    """L'application branchée sur la stack : PostgreSQL (compte app_writer) et registre MLflow."""
    from tennis_mlops.api.app import create_app
    from tennis_mlops.ingestion.store import IngestionStore
    from tennis_mlops.registry.loader import AliasModelLoader
    from tennis_mlops.registry.model_registry import ModelRegistry
    from tennis_mlops.serving.features import FeatureStore
    from tennis_mlops.serving.model import ModelHolder
    from tennis_mlops.serving.persistence import ApiStore, AsyncWriter
    from tennis_mlops.serving.service import PredictionService
    from tennis_mlops.training.config import DEFAULT_TRAIN_CONFIG_PATH, TrainConfig

    settings = settings or Settings()
    config = config or ProjectConfig.from_yaml(DEFAULT_CONFIG_PATH)
    train = TrainConfig.from_yaml(DEFAULT_TRAIN_CONFIG_PATH)
    # MLflow injoignable : on veut l'apprendre en secondes, pas après 7 essais espacés (le cache sert entre-temps).
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    store = ApiStore.from_url(settings.api_sqlalchemy_url())
    registry = ModelRegistry.connect(str(settings.mlflow_tracking_uri), train.registered_model)
    service = PredictionService(
        features=FeatureStore(IngestionStore(store.engine), train.features),
        model=ModelHolder(AliasModelLoader(registry)),
        store=store,
        writer=AsyncWriter(store, max_queue=config.api.write_queue, flush_s=config.api.write_flush_s),
        config=config.api,
    )
    token = settings.admin_token.get_secret_value() if settings.admin_token else None
    return create_app(service, admin_token=token)


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.api", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="lance l'API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--workers", type=int, default=1, help="processus uvicorn (chacun charge modèle et historique)")
    outcomes = commands.add_parser("outcomes", help="dernières prédictions et leur résultat réel")
    outcomes.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    settings = settings or Settings()
    if args.command == "serve":
        import uvicorn

        configure_logging()
        if args.workers > 1:
            uvicorn.run("tennis_mlops.api.__main__:build_app", factory=True, host=args.host, port=args.port,
                        workers=args.workers, log_level="info")  # fmt: skip
        else:
            uvicorn.run(build_app(settings), host=args.host, port=args.port, log_level="info")
        return EXIT_OK
    return _outcomes(settings, args.limit)


def _outcomes(settings: Settings, limit: int) -> int:
    import sqlalchemy as sa
    from sqlalchemy.exc import SQLAlchemyError

    query = sa.text(
        "SELECT created_at, match_id, p1_name, p2_name, p1_win_probability, model_version, duration_ms, client, "
        "actual_winner_id, p1_id, correct FROM prediction_outcomes ORDER BY created_at DESC LIMIT :n"
    ).bindparams(n=limit)
    try:
        with sa.create_engine(settings.sqlalchemy_url()).connect() as conn:
            rows = conn.execute(query).all()
    except SQLAlchemyError as exc:
        print(f"lecture impossible ({type(exc).__name__}) : la stack tourne-t-elle, l'API a-t-elle démarré ?",
              file=sys.stderr)  # fmt: skip
        return EXIT_FAILED
    if not rows:
        print("aucune prédiction pour l'instant")
    for r in rows:
        when = r.created_at.astimezone().strftime("%d/%m %H:%M:%S") if r.created_at.tzinfo else str(r.created_at)
        result = "en attente" if r.actual_winner_id is None else ("juste" if r.correct else "faux")
        duration = f"{r.duration_ms:.0f} ms" if r.duration_ms is not None else "—"
        print(f"{when}  v{r.model_version:<4} {duration:>7}  {result:<10} {r.p1_name or '?'} contre {r.p2_name or '?'} "
              f"({r.p1_win_probability:.0%})  {r.match_id or ''}  [{r.client or '—'}]")  # fmt: skip
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
