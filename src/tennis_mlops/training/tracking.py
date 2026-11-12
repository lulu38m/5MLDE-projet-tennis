"""Accès à MLflow : module technique, sans logique de ML (convention du TP03).

- **Suivi explicite** : chaque appel reçoit le `run_id`. Les candidats sont entraînés en parallèle dans des threads :
  aucun run « actif » implicite n'est partagé entre eux.
- **Pas de dossier `mlruns` local** : le serveur est obligatoire ; s'il ne répond pas, la campagne s'arrête avant
  tout calcul avec une erreur explicite (aucun modèle n'est annoncé comme publié).
- **Aucun run ne reste `RUNNING`** : une étape qui échoue ferme les runs ouverts en `FAILED`, avec l'étape et la cause.
- Les artefacts sont envoyés au serveur (`--serve-artifacts`) : rien n'est écrit sur un chemin du poste client.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import threading
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import mlflow
import pandas as pd
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities import DatasetInput, InputTag, Metric, Param, Run, RunTag
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from sklearn.pipeline import Pipeline

MAX_PARAMS_PER_CALL = 100
MAX_PARAM_LENGTH = 6000
MAX_METRICS_PER_CALL = 1000


class TrackingError(RuntimeError):
    """MLflow est injoignable ou refuse une écriture : la campagne ne peut rien publier."""


class MlflowTracker:
    """Une expérience MLflow et les runs ouverts par la campagne en cours."""

    def __init__(
        self,
        tracking_uri: str,
        experiment: str,
        check_server: bool = True,
        pip_requirements: Sequence[str] | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri.rstrip("/")
        # None : MLflow exporte les dépendances exactes depuis uv.lock (`uv export`) ; une liste fixe sert aux tests.
        self.pip_requirements = list(pip_requirements) if pip_requirements is not None else None
        if check_server:
            self._check_server()
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri)
        existing = self.client.get_experiment_by_name(experiment)
        # Emplacement des artefacts choisi par le serveur : jamais un chemin du poste client.
        self.experiment_id: str = existing.experiment_id if existing else self.client.create_experiment(experiment)
        # L'UI MLflow 3 ouvre alors directement la liste des runs d'entraînement (et non la vue GenAI / traces).
        self.client.set_experiment_tag(self.experiment_id, "mlflow.experimentKind", "custom_model_development")
        self.experiment = experiment
        self._open: set[str] = set()
        self._lock = threading.Lock()

    def _check_server(self) -> None:
        if not self.tracking_uri.startswith("http"):
            return
        try:
            httpx.get(f"{self.tracking_uri}/health", timeout=5).raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"serveur MLflow indisponible ({self.tracking_uri}) : {exc} — `make up` puis relancer"
            raise TrackingError(msg) from exc

    # ------------------------------------------------------------------ runs

    def start_run(self, name: str, tags: Mapping[str, str], parent_id: str | None = None) -> str:
        """Crée un run (enfant si `parent_id`) et le note comme ouvert."""
        all_tags = dict(tags)
        if parent_id:
            all_tags["mlflow.parentRunId"] = parent_id
        run = self.client.create_run(self.experiment_id, run_name=name, tags=all_tags)
        run_id: str = run.info.run_id
        with self._lock:
            self._open.add(run_id)
        return run_id

    def finish(self, run_id: str) -> None:
        """Ferme un run avec succès."""
        self.client.set_terminated(run_id, "FINISHED")
        with self._lock:
            self._open.discard(run_id)

    def fail_run(self, run_id: str, step: str, error: BaseException) -> None:
        """Ferme un run en `FAILED` avec l'étape et la cause (les autres runs continuent)."""
        with self._lock:
            if run_id not in self._open:
                return
            self._open.discard(run_id)
        self.client.set_tag(run_id, "failure.step", step)
        self.client.set_tag(run_id, "failure.cause", f"{type(error).__name__}: {error}"[:5000])
        self.client.set_terminated(run_id, "FAILED")

    def fail_open_runs(self, step: str, error: BaseException) -> list[str]:
        """Ferme en `FAILED` tous les runs encore ouverts, avec l'étape et la cause ; renvoie leurs ids."""
        with self._lock:
            open_runs, self._open = sorted(self._open), set()
        for run_id in open_runs:
            try:
                self.client.set_tag(run_id, "failure.step", step)
                self.client.set_tag(run_id, "failure.cause", f"{type(error).__name__}: {error}"[:5000])
                self.client.set_terminated(run_id, "FAILED")
            except MlflowException:  # serveur tombé entre-temps : le run sera à réconcilier, l'erreur d'origine prime
                continue
        return open_runs

    def run_url(self, run_id: str) -> str:
        """Lien direct vers le run dans l'UI MLflow."""
        return f"{self.tracking_uri}/#/experiments/{self.experiment_id}/runs/{run_id}"

    # ------------------------------------------------------------------ écritures

    def log_params(self, run_id: str, params: Mapping[str, object]) -> None:
        """Paramètres (par paquets de 100, valeurs tronquées à la limite MLflow)."""
        items = [Param(k, str(v)[:MAX_PARAM_LENGTH]) for k, v in params.items()]
        for start in range(0, len(items), MAX_PARAMS_PER_CALL):
            self.client.log_batch(run_id, params=items[start : start + MAX_PARAMS_PER_CALL])

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> list[str]:
        """Métriques finies ; renvoie (et trace en tag) celles qui ne l'étaient pas, plutôt que de les perdre."""
        now = _now_ms()
        finite = [Metric(k, float(v), now, 0) for k, v in metrics.items() if math.isfinite(v)]
        skipped = sorted(k for k, v in metrics.items() if not math.isfinite(v))
        for start in range(0, len(finite), MAX_METRICS_PER_CALL):
            self.client.log_batch(run_id, metrics=finite[start : start + MAX_METRICS_PER_CALL])
        if skipped:
            self.client.set_tag(run_id, "metrics.not_finite", ", ".join(skipped)[:5000])
        return skipped

    def log_series(self, run_id: str, key: str, points: Sequence[tuple[int, float] | tuple[int, float, int]]) -> None:
        """Une métrique suivie pas à pas : (pas, valeur) ou (pas, valeur, horodatage ms) ; MLflow en fait une courbe."""
        now = _now_ms()
        entries = [
            Metric(key, float(p[1]), int(p[2]) if len(p) == 3 else now, int(p[0]))
            for p in points
            if math.isfinite(p[1])
        ]
        for start in range(0, len(entries), MAX_METRICS_PER_CALL):
            self.client.log_batch(run_id, metrics=entries[start : start + MAX_METRICS_PER_CALL])

    def set_tags(self, run_id: str, tags: Mapping[str, object]) -> None:
        """Tags du run."""
        self.client.log_batch(run_id, tags=[RunTag(k, str(v)[:5000]) for k, v in tags.items()])

    def log_text(self, run_id: str, text: str, path: str) -> None:
        """Un artefact texte (CSV, Markdown, YAML…)."""
        self.client.log_text(run_id, text, path)

    def log_dict(self, run_id: str, data: Mapping[str, Any], path: str) -> None:
        """Un artefact JSON."""
        self.client.log_dict(run_id, dict(data), path)

    def log_frame(self, run_id: str, frame: pd.DataFrame, path: str) -> None:
        """Un DataFrame en CSV."""
        self.client.log_text(run_id, frame.to_csv(index=False), path)

    def log_figure(self, run_id: str, figure: Figure, path: str) -> None:
        """Un graphique matplotlib (PNG)."""
        self.client.log_figure(run_id, figure, path, save_kwargs={"dpi": 120, "bbox_inches": "tight"})

    def log_file(self, run_id: str, local_path: Path, artifact_dir: str | None = None) -> None:
        """Un fichier local tel quel."""
        self.client.log_artifact(run_id, str(local_path), artifact_dir)

    def log_parquet(self, run_id: str, frame: pd.DataFrame, path: str) -> str:
        """Un DataFrame en Parquet ; renvoie son SHA-256 (empreinte du fichier, distincte du digest MLflow)."""
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / Path(path).name
            frame.to_parquet(local, index=False)
            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            self.client.log_artifact(run_id, str(local), str(Path(path).parent) if "/" in path else None)
        return digest

    def log_dataset(self, run_id: str, frame: pd.DataFrame, name: str, context: str) -> str:
        """Associe un jeu au run (`mlflow.log_input`) avec son contexte ; renvoie le digest MLflow."""
        with (
            warnings.catch_warnings()
        ):  # schéma du jeu (identifiants entiers) : rien à voir avec la signature du modèle
            warnings.simplefilter("ignore", UserWarning)
            dataset = from_pandas(frame, name=name, targets="label")
            entity = dataset._to_mlflow_entity()
            digest = str(dataset.digest)
        self.client.log_inputs(run_id, [DatasetInput(entity, tags=[InputTag("mlflow.data.context", context)])])
        return digest

    def log_model(
        self,
        run_id: str,
        model: Pipeline,
        signature: ModelSignature,
        input_example: pd.DataFrame,
        trusted_types: Sequence[str],
    ) -> ModelInfo:
        """Journalise le pipeline complet (format skops) dans le run, qui est ensuite fermé avec succès.

        `log_model` passe par l'API « fluide » : le run est repris dans le thread courant (pile de runs propre à
        chaque thread), le temps de cet appel. C'est donc la dernière écriture d'un run de candidat.
        """
        with self._resumed(run_id):
            info: ModelInfo = mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="skops",
                skops_trusted_types=list(trusted_types),
                signature=signature,
                input_example=input_example,
                pyfunc_predict_fn="predict_proba",
                pip_requirements=self.pip_requirements,
            )
        with self._lock:
            self._open.discard(run_id)
        return info

    @contextmanager
    def _resumed(self, run_id: str) -> Iterator[None]:
        with mlflow.start_run(run_id=run_id):
            yield

    # ------------------------------------------------------------------ lectures

    def previous_campaign(self, exclude_run_id: str) -> Run | None:
        """Dernière campagne terminée avec succès (pour mesurer la dérive des données)."""
        runs = self.client.search_runs(
            [self.experiment_id],
            filter_string="tags.kind = 'campaign' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=5,
        )
        return next((r for r in runs if r.info.run_id != exclude_run_id), None)

    def read_json_artifact(self, run_id: str, path: str) -> dict[str, Any] | None:
        """Contenu d'un artefact JSON, ou None s'il n'existe pas."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                local = self.client.download_artifacts(run_id, path, tmp)
            except (MlflowException, OSError):
                return None
            data: dict[str, Any] = json.loads(Path(local).read_text(encoding="utf-8"))
        return data


# --------------------------------------------------------------------------- reproductibilité


IMAGE_COMMIT_ENV = "TENNIS_GIT_COMMIT"  # posé par le Dockerfile (`make up` passe `git rev-parse HEAD`)


def code_info(root: Path) -> dict[str, str]:
    """Révision Git (et état propre / modifié), Python, empreinte de `uv.lock` : de quoi rejouer la campagne."""
    info = {
        "python.version": platform.python_version(),
        "platform": platform.platform(terse=True),
        "mlflow.source.name": "tennis_mlops.flows.train",
        "mlflow.source.type": "LOCAL",
    }
    lock = root / "uv.lock"
    info["uv_lock.sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else "absent"
    commit = _git(root, "rev-parse", "HEAD")
    if commit:
        info["git.dirty"] = str(bool(_git(root, "status", "--porcelain", "--untracked-files=no")))
    elif commit := os.environ.get(IMAGE_COMMIT_ENV, "").strip():  # worker Docker : révision gravée au build
        info["git.dirty"] = "inconnu (image Docker)"
    info["git.commit"] = commit or "inconnu (pas de dépôt Git)"
    if commit:
        info["mlflow.source.git.commit"] = commit
    return info


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _now_ms() -> int:
    return int(time.time() * 1000)
