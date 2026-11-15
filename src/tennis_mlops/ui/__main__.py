"""`python -m tennis_mlops.ui [--port 8501] [--host 127.0.0.1]` : lance l'interface Streamlit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP = Path(__file__).with_name("app.py")
THEME = {"primaryColor": "#c0522b", "backgroundColor": "#fbfaf7", "secondaryBackgroundColor": "#f1ede4",
         "textColor": "#1f2a24"}  # fmt: skip


def command(host: str, port: int) -> list[str]:
    """La ligne de commande Streamlit : pas de télémétrie, pas de rechargement à chaud, thème terre battue."""
    args = ["streamlit", "run", str(APP), "--server.address", host, "--server.port", str(port),
            "--server.headless", "true", "--browser.gatherUsageStats", "false", "--server.runOnSave", "false",
            "--server.fileWatcherType", "none", "--client.toolbarMode", "minimal"]  # fmt: skip
    for key, value in THEME.items():
        args += [f"--theme.{key}", value]
    return args


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.ui", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)
    from streamlit.web import cli

    sys.argv = command(args.host, args.port)
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
