"""Experiments: one entry per model name, merging local trainings and MLflow history."""
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

import mlflow
from fastapi import BackgroundTasks, UploadFile
from mlflow.entities import Experiment as MlflowExperiment

from src.config import get_settings
from src.db.store import get_store
from src.integrations.k8s import delete_training_job, job_name_for
from src.integrations.minio_storage import delete_training_data, upload_csv_to_minio
from src.integrations.mlflow_tracking import (
    TRAINING_ID_TAG,
    get_active_experiment,
    prepare_experiment_for_training,
)
from src.schemas import Experiment, PaginatedResult
from src.services.pagination import paginate
from src.training.launcher import train_model_job

REASON_NO_SUCCESSFUL_RUN = "No successful run in MLflow"


@dataclass
class RunSummary:
    """What the experiment listing needs to know about the runs of one MLflow experiment."""
    latest_status: Optional[str] = None
    latest_start: int = -1
    has_finished: bool = False
    finished_training_ids: set = field(default_factory=set)

    def add(self, status: str, start_time: Optional[int], training_id: Optional[str]) -> None:
        start = int(start_time or 0)
        if start >= self.latest_start:
            self.latest_start = start
            self.latest_status = status
        if status == "FINISHED":
            self.has_finished = True
            if training_id:
                self.finished_training_ids.add(training_id)


# ---- status resolution (local training vs MLflow) ----------------------------


def resolve_status(training: Optional[Dict[str, Any]], summary: Optional[RunSummary]) -> tuple:
    """Single source of truth for the experiment status. Returns (status, status_reason).

    - The latest local training decides while it is active, failed or completed...
    - ...except that a FINISHED MLflow run of that training means it is completed.
    - Without local trainings, MLflow decides: RUNNING -> training, any FINISHED -> completed.
    """
    if training is not None:
        if training["status"] == "cancelled":
            # Only visible while the cleanup of a cancellation is pending a retry.
            return "cancelled", training.get("status_reason")
        if summary is not None and training["training_id"] in summary.finished_training_ids:
            return "completed", None
        return training["status"], training.get("status_reason")

    if summary is None:
        return "failed", REASON_NO_SUCCESSFUL_RUN
    if summary.latest_status in ("RUNNING", "SCHEDULED"):
        return "training", None
    if summary.has_finished:
        return "completed", None
    return "failed", REASON_NO_SUCCESSFUL_RUN


def build_experiment(
    name: str,
    training: Optional[Dict[str, Any]],
    mlflow_exp: Optional[MlflowExperiment],
    summary: Optional[RunSummary],
) -> Experiment:
    status, reason = resolve_status(training, summary)
    t = training or {}
    return Experiment(
        experiment_name=name,
        source="mlflow" if mlflow_exp is not None else "local",
        status=status,
        status_reason=reason,
        internal_id=t.get("training_id"),
        mlflow_id=mlflow_exp.experiment_id if mlflow_exp is not None else None,
        artifact_location=mlflow_exp.artifact_location if mlflow_exp is not None else None,
        model=t.get("model_type"),
        model_name=t.get("model_name"),
        config=t.get("config"),
        dependencies=t.get("dependencies") or [],
        csv_filename=t.get("csv_filename"),
        created_at=t.get("created_at"),
    )


def summarize_runs(experiment_ids: List[str]) -> Dict[str, RunSummary]:
    summaries: Dict[str, RunSummary] = {eid: RunSummary() for eid in experiment_ids}
    if not experiment_ids:
        return summaries
    runs = mlflow.search_runs(experiment_ids=experiment_ids, output_format="list")
    for run in cast(list, runs):
        summary = summaries.setdefault(run.info.experiment_id, RunSummary())
        summary.add(run.info.status, run.info.start_time, run.data.tags.get(TRAINING_ID_TAG))
    return summaries


# ---- queries -----------------------------------------------------------------


def get_all_experiments(mlflow_on: bool, page: int = 0, pageSize: int = 20, search: str = '') -> PaginatedResult[Experiment]:
    # Latest training per experiment name (list_all is newest first).
    latest: Dict[str, Dict[str, Any]] = {}
    for training in get_store().list_all():
        latest.setdefault(training["experiment_name"], training)

    mlflow_exps: Dict[str, MlflowExperiment] = {}
    summaries: Dict[str, RunSummary] = {}
    if mlflow_on:
        try:
            for exp in mlflow.search_experiments():
                if exp.name != "Default":
                    mlflow_exps[exp.name] = exp
            summaries = summarize_runs([e.experiment_id for e in mlflow_exps.values()])
        except Exception as e:
            print(f"Error reading from MLflow: {str(e)}")

    all_experiments: List[Experiment] = []
    # Experiments with local trainings first (newest first), then MLflow-only ones.
    for name, training in latest.items():
        exp = mlflow_exps.get(name)
        summary = summaries.get(exp.experiment_id) if exp is not None else None
        all_experiments.append(build_experiment(name, training, exp, summary))
    for name, exp in mlflow_exps.items():
        if name not in latest:
            all_experiments.append(build_experiment(name, None, exp, summaries.get(exp.experiment_id)))

    if search:
        search_lower = search.lower()
        all_experiments = [exp for exp in all_experiments if search_lower in exp["experiment_name"].lower()]

    return paginate(all_experiments, page, pageSize)


async def get_experiment(experiment_name: str) -> Experiment:
    training = get_store().latest(experiment_name)
    mlflow_exp = get_active_experiment(experiment_name) if get_settings().mlflow_enabled else None

    if mlflow_exp is None and training is None:
        # Neither in MLflow nor in the local store: 404
        raise KeyError()

    summary = None
    if mlflow_exp is not None:
        summary = summarize_runs([mlflow_exp.experiment_id]).get(mlflow_exp.experiment_id)
    return build_experiment(experiment_name, training, mlflow_exp, summary)


# ---- commands ----------------------------------------------------------------


async def create_experiment(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile,
    model_type: str,
    experiment_name: str,
    model_name: str,
    config_dict: dict[str, Any],
    dependencies_list: List[Any],
) -> Experiment:
    """Registers a training and launches its Kubernetes Job in the background.
    Raises ActiveTrainingExists if the experiment already has a training in progress."""
    store = get_store()
    file_content = await csv_file.read()
    training_id = str(uuid.uuid4())[:8]
    filename = str(csv_file.filename if csv_file.filename is not None else "csvfile")

    # 1. Reserve the training first: the database rejects a second active training
    #    for the same experiment, before anything is uploaded.
    training = store.insert({
        "training_id": training_id,
        "experiment_name": experiment_name,
        "status": "queued",
        "model_type": model_type,
        "model_name": model_name,
        "config": config_dict,
        "dependencies": dependencies_list,
        "csv_filename": filename,
        "job_name": job_name_for(training_id),
    })

    # 2. Upload the dataset to MinIO
    try:
        minio_path = upload_csv_to_minio(file_content, training_id, filename)
        store.update(training_id, minio_path=minio_path)
    except Exception:
        store.delete(training_id)
        raise

    # 3. A training cannot log into a soft-deleted MLflow experiment with the same name
    if get_settings().mlflow_enabled:
        try:
            prepare_experiment_for_training(experiment_name)
        except Exception as e:
            print(f"[{training_id}] Could not prepare MLflow experiment '{experiment_name}': {e}")

    # 4. Delegate the execution to Kubernetes as a background task
    background_tasks.add_task(
        train_model_job,
        training_id=training_id,
        model_type=model_type,
        config=config_dict,
        minio_path=minio_path,
        experiment_name=experiment_name,
        model_name=model_name
    )

    return build_experiment(experiment_name, store.get(training_id) or training, None, None)


def delete_experiment(experiment_name: str, mlflowb: bool) -> bool:
    """Deletes the experiment (local rows, MLflow and MinIO datasets), stopping the Job of any training in progress.

    The training CSVs of every training of the experiment are deleted from MinIO (a training
    without objects is not an error). If a Job cannot be deleted, the ApiException propagates;
    if MinIO fails, its error propagates. In both cases nothing else is deleted (rows and
    MLflow experiment are kept), so the call can be retried.
    """
    store = get_store()
    for training in store.active(experiment_name):
        job_name = training.get("job_name") or job_name_for(training["training_id"])
        # A Job that no longer exists counts as stopped (delete_training_job returns False).
        if delete_training_job(job_name):
            print(f"Kubernetes: job {job_name} of experiment '{experiment_name}' deleted.")

    for training in store.list_by_name(experiment_name):
        removed = delete_training_data(training["training_id"])
        print(f"MinIO: {removed} object(s) of training {training['training_id']} deleted.")

    found = False
    deleted = store.delete_by_name(experiment_name)
    if deleted:
        found = True
        print(f"Local DB: {deleted} training(s) of experiment '{experiment_name}' deleted.")

    if mlflowb:
        try:
            # Buscar el experimento en MLflow por su nombre
            mlflow_exp = mlflow.get_experiment_by_name(experiment_name)

            # Si existe y no está ya borrado, lo eliminamos
            if mlflow_exp and mlflow_exp.lifecycle_stage != "deleted":
                mlflow.delete_experiment(mlflow_exp.experiment_id)
                found = True
                print(f"MLflow: Experiment '{experiment_name}' deleted.")
        except Exception as e:
            print(f"Error deleting from MLflow: {str(e)}")

    return found
