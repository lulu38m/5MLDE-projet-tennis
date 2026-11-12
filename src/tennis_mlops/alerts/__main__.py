"""`python -m tennis_mlops.alerts preview|test|serve|recent` : cartes Teams et service d'alertes.

- `preview` : écrit le JSON des cartes d'exemple dans `state/alerts/` (à coller dans l'Adaptive Cards Designer).
- `test [--example quarantaine]` : envoie une seule carte, marquée « 🧪 TEST », au webhook de `.env`
  (`make alert-test`). Code de sortie 0 si Teams l'a acceptée, 1 si l'envoi a échoué, 2 si aucun webhook.
- `serve` : le service `alert_sink` (port 8090) qui reçoit les automations Prefect et les décisions du registre.
- `recent [--limit 15]` : les dernières alertes et ce qu'elles sont devenues (`make alerts`), lues dans la base.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tennis_mlops.alerts.cards import teams_message
from tennis_mlops.alerts.examples import EXAMPLES
from tennis_mlops.alerts.teams import Delivery, TeamsNotifier
from tennis_mlops.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, ProjectConfig, Settings

EXIT_SENT, EXIT_FAILED, EXIT_NOT_CONFIGURED = 0, 1, 2
PREVIEW_DIR = PROJECT_ROOT / "state" / "alerts"
SINK_PORT = 8090
DELIVERY_LABELS = {"sent": "envoyée", "filtered": "filtrée", "repeated": "répétée", "disabled": "sans webhook",
                   "failed": "ÉCHEC"}  # fmt: skip


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.alerts", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="JSON des cartes d'exemple")
    preview.add_argument("--out", type=Path, default=PREVIEW_DIR)
    test = commands.add_parser("test", help="envoie UNE carte de test au webhook Teams")
    test.add_argument("--example", choices=sorted(EXAMPLES), default="quarantaine")
    serve = commands.add_parser("serve", help="service d'alertes (alert_sink)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=SINK_PORT)
    recent = commands.add_parser("recent", help="dernières alertes (base du projet)")
    recent.add_argument("--limit", type=int, default=15)
    args = parser.parse_args(argv)
    settings = settings or Settings()

    if args.command == "serve":
        return _serve(settings, args.host, args.port)
    if args.command == "recent":
        return _recent(settings, args.limit)

    if args.command == "preview":
        args.out.mkdir(parents=True, exist_ok=True)
        for name, make in EXAMPLES.items():
            (args.out / f"{name}.json").write_text(json.dumps(teams_message(make()), ensure_ascii=False, indent=2),
                                                   encoding="utf-8")  # fmt: skip
        print(f"{len(EXAMPLES)} cartes écrites dans {args.out} (Adaptive Cards Designer : coller le champ « content »)")
        return EXIT_SENT

    example = EXAMPLES[args.example]()
    alert = dataclasses.replace(
        example,
        title=f"🧪 TEST · {example.title.split(' ', 1)[1]}",
        status="Message de test (make alert-test) : aucun incident réel",
        at=datetime.now(UTC),
    )
    result = TeamsNotifier(settings.teams_webhook_url).send(alert)
    if result.status is Delivery.DISABLED:
        print("Aucun webhook : ajouter TENNIS_TEAMS_WEBHOOK_URL=… dans .env (jamais dans Git).", file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    if result.status is Delivery.FAILED:
        print(f"Envoi refusé : {result.detail}", file=sys.stderr)
        return EXIT_FAILED
    print(f"Carte de test « {args.example} » envoyée sur Teams ({result.detail}).")
    return EXIT_SENT


def _serve(settings: Settings, host: str, port: int) -> int:
    import os

    import uvicorn

    from tennis_mlops.alerts.prefect_reader import PrefectFlowRunReader
    from tennis_mlops.alerts.sink import AlertService, MlflowCampaigns, card_context_factory, create_app
    from tennis_mlops.ingestion.store import IngestionStore
    from tennis_mlops.orchestration.events import PipelineEvents
    from tennis_mlops.training.config import DEFAULT_TRAIN_CONFIG_PATH, TrainConfig

    # Une carte ne doit pas attendre MLflow plus de quelques secondes (sans cela : 5 essais espacés).
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    config = ProjectConfig.from_yaml(DEFAULT_CONFIG_PATH).orchestration
    store = IngestionStore.from_url(settings.sqlalchemy_url())
    service = AlertService(
        events=PipelineEvents(store.engine),
        reader=PrefectFlowRunReader(),
        notifier=TeamsNotifier(settings.teams_webhook_url),
        context=card_context_factory(
            settings.alert_origin(), settings.alert_owner, str(settings.prefect_ui_url), str(settings.mlflow_ui_url)
        ),
        teams_events=frozenset(config.teams_events),
        cooldown=timedelta(minutes=config.alert_cooldown_min),
        store=store,
        campaigns=MlflowCampaigns(
            str(settings.mlflow_tracking_uri), TrainConfig.from_yaml(DEFAULT_TRAIN_CONFIG_PATH).experiment
        ),
    )
    teams = "Teams activé" if service.notifier.enabled else "aucun webhook Teams (cartes gardées dans l'historique)"
    print(f"alert_sink : {teams} ; envoyés : {', '.join(config.teams_events) or 'aucun'}", flush=True)
    uvicorn.run(create_app(service), host=host, port=port, log_level="info")
    return EXIT_SENT


def _recent(settings: Settings, limit: int) -> int:
    from sqlalchemy.exc import SQLAlchemyError

    from tennis_mlops.orchestration.events import EventKind, PipelineEvents

    try:
        rows = PipelineEvents.from_url(settings.sqlalchemy_url()).recent(limit=limit, kinds=(EventKind.ALERT,))
    except SQLAlchemyError as exc:
        print(f"base injoignable ({type(exc).__name__}) : la stack tourne-t-elle ? (make up)", file=sys.stderr)
        return EXIT_FAILED
    if not rows:
        print("aucune alerte pour l'instant")
    for row in rows:
        delivery = DELIVERY_LABELS.get(row["payload"].get("delivery"), row["payload"].get("delivery", "?"))
        when = row["occurred_at"].astimezone().strftime("%d/%m %H:%M:%S")
        print(f"{when}  {delivery:<12}  {row['title']}")
        if row["payload"].get("delivery") == "failed":
            print(f"{'':16}{row['payload'].get('detail', '')}")
    return EXIT_SENT


if __name__ == "__main__":
    raise SystemExit(main())
