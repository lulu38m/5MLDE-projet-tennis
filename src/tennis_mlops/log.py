"""Configuration commune des logs (format lisible en terminal comme dans les logs Prefect)."""

import logging
import warnings

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s — %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure le logger racine une seule fois."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "filelock", "great_expectations"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def quiet_training_noise() -> None:
    """Retire du terminal deux avertissements tiers sans conséquence, répétés à chaque candidat.

    - SciPy 1.18 signale une option que scikit-learn 1.6 passe encore à son solveur (`iprint`) ;
    - MLflow ne trouve pas la version de pip (absent d'un environnement uv) pour son `conda.yaml` : les
      dépendances réelles du modèle viennent de `uv.lock` (`requirements.txt` du modèle).
    """
    warnings.filterwarnings("ignore", message="Unknown solver options", module="sklearn")
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
