"""`python -m tennis_mlops.acceptance bootstrap|run|urls` : le parcours du correcteur.

- `bootstrap` : fin de `make up` — premier modèle entraîné par la chaîne et servi par l'API (rien à faire s'il y en a
  déjà un) ;
- `run [--only S2,S7] [--teams] [--force]` : `make scenarios` — les tests d'acceptation S1 → S9, rapport HTML dans
  `state/acceptance/rapport.html` ; code de sortie 0 si tout est vert, 1 sinon, 2 si la stack n'est pas prête ;
- `urls` : `make urls` — les interfaces, leurs identifiants de démo, et si elles répondent.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence

from tennis_mlops.config import PROJECT_ROOT, Settings

EXIT_OK, EXIT_FAILED, EXIT_NOT_READY = 0, 1, 2


def configure_environment(settings: Settings) -> None:
    """Avant d'importer Prefect et MLflow : où est Prefect, et des logs de terminal lisibles.

    Les flows lancés par les scénarios gardent leurs logs complets dans l'UI Prefect. Le terminal, lui, n'affiche que
    les étapes vérifiées : les incidents provoqués exprès (lot refusé, source en panne) y écriraient sinon des
    traces d'erreur qui sont justement le comportement attendu.
    """
    os.environ.setdefault("PREFECT_API_URL", str(settings.prefect_api_url))
    os.environ.setdefault("PREFECT_LOGGING_HANDLERS_CONSOLE_LEVEL", "CRITICAL")
    os.environ.setdefault("PREFECT_LOGGING_HANDLERS_CONSOLE_FLOW_RUNS_LEVEL", "CRITICAL")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "ERROR")
    for name in ("mlflow", "streamlit", "alembic"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.acceptance", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bootstrap", help="premier modèle en production (fin de make up)")
    run = commands.add_parser("run", help="tests d'acceptation S1 → S9 (make scenarios)")
    run.add_argument("--only", default="", help="scénarios à jouer, séparés par des virgules (ex. S2,S5)")
    run.add_argument("--teams", action="store_true", help="laisser partir les cartes Teams (pas de mode silence)")
    run.add_argument("--force", action="store_true", help="jouer même hors de l'état de référence de make up")
    run.add_argument("--report-dir", default=None, help="dossier du rapport (défaut : state/acceptance)")
    commands.add_parser("urls", help="interfaces de la stack (make urls)")
    args = parser.parse_args(argv)
    settings = Settings()

    if args.command == "urls":
        from tennis_mlops.acceptance.urls import lines

        print("\n".join(lines(settings)))
        return EXIT_OK
    if args.command == "bootstrap":
        configure_environment(settings)
        return _bootstrap(settings)
    from tennis_mlops.acceptance.scenarios import scenario_ids

    only = frozenset(s.strip().upper() for s in args.only.split(",") if s.strip())
    unknown = sorted(only - set(scenario_ids()))
    if unknown:
        print(f"scénarios inconnus : {', '.join(unknown)} (S1 … S9)", file=sys.stderr)
        return EXIT_NOT_READY
    configure_environment(settings)
    return _run(settings, only, teams=args.teams, force=args.force, report_dir=args.report_dir)


def _bootstrap(settings: Settings) -> int:
    import httpx
    import sqlalchemy as sa

    from tennis_mlops.acceptance.bootstrap import BootstrapError, bootstrap
    from tennis_mlops.acceptance.scenarios import LocalActions
    from tennis_mlops.acceptance.stack import Stack, WaitTimeoutError

    stack = Stack(settings=settings)
    try:
        bootstrap(stack, LocalActions(PROJECT_ROOT))
    except (BootstrapError, WaitTimeoutError, httpx.HTTPError, sa.exc.SQLAlchemyError) as exc:
        print(f"❌ premier modèle : {exc}", file=sys.stderr)
        return EXIT_FAILED
    finally:
        stack.close()
    print(
        "stack prête : `uv run tennis urls` (adresses), `uv run tennis scenarios` (tests d'acceptation), "
        "`uv run tennis demo` (la chaîne seule) ; ou make urls, make scenarios, make demo"
    )
    return EXIT_OK


def _run(settings: Settings, only: frozenset[str], *, teams: bool, force: bool, report_dir: str | None) -> int:
    from pathlib import Path

    from tennis_mlops.acceptance.runner import REPORT_DIR, Campaign, NotReadyError
    from tennis_mlops.acceptance.scenarios import LocalActions
    from tennis_mlops.acceptance.services import ComposeServices
    from tennis_mlops.acceptance.stack import Stack

    stack = Stack(settings=settings)
    campaign = Campaign(stack, ComposeServices(PROJECT_ROOT), LocalActions(PROJECT_ROOT), only=only, teams=teams,
                        force=force)  # fmt: skip
    try:
        result = campaign.run()
    except NotReadyError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return EXIT_NOT_READY
    finally:
        stack.close()
    page = result.write(Path(report_dir) if report_dir else PROJECT_ROOT / REPORT_DIR)
    shown = page.relative_to(PROJECT_ROOT) if page.is_relative_to(PROJECT_ROOT) else page
    print(f"\n{result.summary()}\nrapport : {shown}  (uv run tennis report, ou make scenarios-report)")
    return EXIT_OK if result.passed else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
