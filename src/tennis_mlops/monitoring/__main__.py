"""`python -m tennis_mlops.monitoring [--port 9101]` : l'exporteur Prometheus de la chaîne (service `exporter`).

Si le socket Docker est monté, il publie aussi le CPU et la mémoire de chaque conteneur du projet (`containers.py`).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import httpx
from prometheus_client import CollectorRegistry, start_http_server

from tennis_mlops.config import Settings
from tennis_mlops.monitoring.containers import SOCKET, ContainerCollector
from tennis_mlops.monitoring.exporter import ChainCollector, Targets


def targets(settings: Settings) -> Targets:
    """Les adresses de la configuration (internes dans la stack, celles de `.env` sur la machine hôte)."""

    def clean(url: object) -> str:
        return str(url).rstrip("/")

    return Targets(mlflow=clean(settings.mlflow_tracking_uri), mock_source=clean(settings.mock_source_url),
                   prefect_api=clean(settings.prefect_api_url), api=clean(settings.api_url),
                   alert_sink=clean(settings.alert_sink_url))  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    """Sert `/metrics` jusqu'à l'arrêt du processus."""
    parser = argparse.ArgumentParser(prog="python -m tennis_mlops.monitoring", description=__doc__)
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--docker-socket", default=SOCKET, help="socket de l'API Docker (CPU et mémoire)")
    parser.add_argument("--compose-project", default="tennis-mlops", help="conteneurs suivis (label compose)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # une ligne par requête, à chaque collecte : trop bavard
    log = logging.getLogger(__name__)
    registry = CollectorRegistry()
    registry.register(ChainCollector(targets(Settings())))
    if Path(args.docker_socket).exists():
        docker = httpx.Client(transport=httpx.HTTPTransport(uds=args.docker_socket), base_url="http://docker",
                              timeout=5.0)  # fmt: skip
        registry.register(ContainerCollector(args.compose_project, docker))
        log.info("conteneurs du projet %s suivis par %s", args.compose_project, args.docker_socket)
    else:
        log.warning("pas de socket Docker (%s) : ni CPU ni mémoire par conteneur", args.docker_socket)
    start_http_server(args.port, registry=registry)
    log.info("exporteur en écoute sur :%s/metrics", args.port)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
