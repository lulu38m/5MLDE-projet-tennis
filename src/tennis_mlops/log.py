"""Configuration commune des logs (format lisible en terminal comme dans les logs Prefect)."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s — %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure le logger racine une seule fois."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "filelock", "great_expectations"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
