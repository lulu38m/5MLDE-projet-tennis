"""`uv run tennis …` : le parcours correcteur, identique sous macOS, Linux et Windows (PowerShell).

Seuls prérequis : Docker Desktop, uv et git. Aucune commande Unix ici (ni `cp`, ni `rm`, ni `make`) : uniquement
Python, `docker compose` et les modules du projet, lancés avec l'interpréteur de l'environnement uv. Le Makefile
appelle les mêmes commandes ; `make up` et `uv run tennis up` font exactement la même chose.

| Commande | Effet |
|---|---|
| `up` | `.env` créé si absent, données vérifiées, stack construite et démarrée, premier modèle servi |
| `urls` | interfaces, identifiants de démo, et si elles répondent |
| `scenarios [--only S2,S5] [--teams] [--force]` | tests d'acceptation S1 → S9, rapport HTML |
| `report` | ouvre le rapport du dernier `scenarios` |
| `demo` / `demo-stop` | la chaîne tourne seule / en pause |
| `down` / `reset` | arrêt (données gardées) / remise à zéro complète (sauf les données téléchargées) |
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from tennis_mlops.config import PROJECT_ROOT

EXIT_OK, EXIT_FAILED = 0, 1
RESET_DIRS = ("state/gx", "state/lots", "state/quarantine", "state/alerts", "state/acceptance")
REPORT = Path("state") / "acceptance" / "rapport.html"
IMAGE = "tennis-mlops:local"

Runner = Callable[..., subprocess.CompletedProcess[str]]


class Cli:
    """Les commandes, avec un lanceur de processus remplaçable (tests)."""

    def __init__(self, root: Path = PROJECT_ROOT, run: Runner = subprocess.run,
                 log: Callable[[str], None] = print) -> None:  # fmt: skip
        self.root = root
        self.run = run
        self.log = log

    # ------------------------------------------------------------------ outils

    def env(self) -> dict[str, str]:
        """Variables passées à `docker compose` et aux modules du projet (ce que le Makefile exporte)."""
        env = dict(os.environ)
        env.setdefault("GIT_COMMIT", self._git_revision())
        env.setdefault("TENNIS_ALERT_HOST", platform.node())
        env.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        env.setdefault("PYTHONUTF8", "1")  # Windows : accents et symboles (✅ ●) dans la console
        if hasattr(os, "getuid"):  # macOS, Linux : le worker écrit dans ./state sous l'identité de l'hôte
            env.setdefault("HOST_UID", str(os.getuid()))
            env.setdefault("HOST_GID", str(os.getgid()))
        if "PREFECT_API_URL" not in env:
            from tennis_mlops.config import Settings

            env["PREFECT_API_URL"] = str(Settings(_env_file=self.root / ".env").prefect_api_url)
        return env

    def call(self, command: Sequence[str], env: Mapping[str, str] | None = None) -> int:
        """Lance une commande dans le dossier du projet, sa sortie dans le terminal ; renvoie son code."""
        try:
            done = self.run(list(command), cwd=self.root, env=dict(env or self.env()), check=False)
        except FileNotFoundError:
            self.log(f"❌ commande introuvable : {command[0]} (installée et dans le PATH ?)")
            return EXIT_FAILED
        return int(done.returncode)

    def module(self, name: str, *args: str) -> int:
        """`python -m <module> …` avec l'interpréteur de l'environnement uv du projet."""
        return self.call([sys.executable, "-m", name, *args])

    def _git_revision(self) -> str:
        try:
            done = self.run(["git", "rev-parse", "HEAD"], cwd=self.root, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return ""
        return str(done.stdout or "").strip()

    def docker_ready(self) -> bool:
        """Docker répond-il ? Sinon, dire quoi faire plutôt que laisser `compose` échouer."""
        try:
            done = self.run(["docker", "info", "--format", "{{.ServerVersion}}"], cwd=self.root,
                            capture_output=True, text=True, check=False)  # fmt: skip
        except FileNotFoundError:
            self.log(
                "❌ docker introuvable : installer Docker Desktop (https://www.docker.com/products/docker-desktop/)"
            )
            return False
        if done.returncode != 0:
            self.log("❌ Docker ne répond pas : ouvrir Docker Desktop, attendre « Engine running », puis relancer")
            return False
        return True

    # ------------------------------------------------------------------ commandes

    def up(self) -> int:
        """`.env`, données, stack, premier modèle."""
        env_file, example = self.root / ".env", self.root / ".env.example"
        if not env_file.exists():
            shutil.copyfile(example, env_file)
            self.log(".env créé depuis .env.example (rien à remplir pour la démo)")
        if not self.docker_ready():
            return EXIT_FAILED
        (self.root / "state").mkdir(exist_ok=True)  # sinon Docker le crée au nom de root (Linux)
        steps: list[Callable[[], int]] = [
            lambda: self.module("tennis_mlops.data", "download"),
            lambda: self.call(["docker", "compose", "up", "-d", "--build", "--wait"]),
            lambda: self.module("tennis_mlops.acceptance", "bootstrap"),
        ]
        for step in steps:
            code = step()
            if code != EXIT_OK:
                self.log("❌ arrêt : voir le message ci-dessus (docker compose ps, docker compose logs <service>)")
                return code
        return EXIT_OK

    def down(self) -> int:
        """La stack s'arrête, les volumes restent."""
        return self.call(["docker", "compose", "down"])

    def reset(self) -> int:
        """Volumes supprimés, état local effacé ; les données téléchargées (`data/raw`) restent."""
        code = self.call(["docker", "compose", "down", "--volumes", "--remove-orphans"])
        if code != EXIT_OK:
            return code
        stubborn = []
        for name in RESET_DIRS:
            try:
                shutil.rmtree(self.root / name)
            except FileNotFoundError:
                continue
            except PermissionError:
                stubborn.append(name)
        if stubborn:  # Linux : fichiers écrits par un conteneur sous un autre utilisateur
            mount = f"{self.root / 'state'}:/app/state"
            code = self.call(["docker", "run", "--rm", "--user", "0", "-v", mount, IMAGE, "rm", "-rf",
                              *(f"/app/{name}" for name in stubborn)])  # fmt: skip
        self.log("stack remise à zéro : `uv run tennis up` pour repartir de l'état de référence")
        return code

    def report(self) -> int:
        """Ouvre le rapport HTML dans le navigateur."""
        page = self.root / REPORT
        if not page.exists():
            self.log("aucun rapport : lancer d'abord `uv run tennis scenarios`")
            return EXIT_FAILED
        self.log(f"rapport : {page}")
        webbrowser.open(page.resolve().as_uri())
        return EXIT_OK


def main(argv: Sequence[str] | None = None, cli: Cli | None = None) -> int:
    """Point d'entrée de `uv run tennis`."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):  # une console Windows sans UTF-8 ne fait pas échouer la commande
        reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="tennis", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("up", help="stack + données + premier modèle servi")
    commands.add_parser("urls", help="interfaces, identifiants, état")
    scenarios = commands.add_parser("scenarios", help="tests d'acceptation S1 → S9")
    scenarios.add_argument("--only", default="", help="ex. S2,S5")
    scenarios.add_argument("--teams", action="store_true", help="laisser partir les cartes Teams")
    scenarios.add_argument("--force", action="store_true", help="jouer hors de l'état de référence")
    commands.add_parser("report", help="ouvre le rapport des tests d'acceptation")
    commands.add_parser("demo", help="la chaîne tourne seule (une semaine simulée par minute)")
    commands.add_parser("demo-stop", help="met la simulation en pause")
    commands.add_parser("down", help="arrête la stack (données gardées)")
    commands.add_parser("reset", help="remise à zéro complète (sauf data/raw)")
    args = parser.parse_args(argv)
    cli = cli or Cli()

    if args.command == "up":
        return cli.up()
    if args.command == "urls":
        return cli.module("tennis_mlops.acceptance", "urls")
    if args.command == "scenarios":
        options = [*(["--only", args.only] if args.only else []), *(["--teams"] if args.teams else []),
                   *(["--force"] if args.force else [])]  # fmt: skip
        return cli.module("tennis_mlops.acceptance", "run", *options)
    if args.command == "report":
        return cli.report()
    if args.command in ("demo", "demo-stop"):
        return cli.module("tennis_mlops.orchestration", "start" if args.command == "demo" else "stop")
    if args.command == "down":
        return cli.down()
    return cli.reset()


if __name__ == "__main__":
    raise SystemExit(main())
