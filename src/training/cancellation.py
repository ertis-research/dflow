"""
Cleanup of cancelled trainings.

Cancelling marks the training as 'cancelled' and then removes everything it created: the
Kubernetes Job (and its pods), its MLflow runs and its dataset in MinIO. The training record is
only deleted when every step succeeded; otherwise it stays as 'cancelled' with the errors in
`status_reason`, and the cleanup is retried (by the status poller, or by calling cancel again),
so nothing is left behind untracked.
"""
from typing import Any, Dict, List

import mlflow

from src.config import get_settings
from src.db.store import get_store
from src.integrations.k8s import delete_training_job, job_name_for
from src.integrations.minio_storage import delete_training_data
from src.integrations.mlflow_tracking import (
    discard_run,
    find_training_runs,
    get_active_experiment,
    has_finished_runs,
)
from src.schemas import ACTIVE_STATUSES

INCOMPLETE_PREFIX = "Cancellation incomplete, it will be retried: "


def cleanup_cancelled_training(training: Dict[str, Any]) -> Dict[str, Any]:
    """Removes what a cancelled training created. Safe to call again after a partial failure.

    Returns {"deleted": {...}, "errors": [...], "experiment_deleted": bool, "cleanup_complete": bool}.
    """
    store = get_store()
    settings = get_settings()
    training_id = training["training_id"]
    experiment_name = training["experiment_name"]

    errors: List[str] = []
    deleted: Dict[str, Any] = {
        "k8s_job": None,
        "mlflow_runs": [],
        "model_versions": [],
        "minio_objects": 0,
        "local_record": False,
    }

    # 1. Kubernetes Job (pods are garbage-collected with it)
    job_name = training.get("job_name") or job_name_for(training_id)
    try:
        if delete_training_job(job_name):
            deleted["k8s_job"] = job_name
    except Exception as e:
        errors.append(f"Kubernetes: could not delete job {job_name}: {e}")

    # 2. MLflow runs of this training: KILLED, then deleted with their model versions
    mlflow_exp = None
    if settings.mlflow_enabled:
        try:
            mlflow_exp = get_active_experiment(experiment_name)
            if mlflow_exp is not None:
                for run in find_training_runs(mlflow_exp.experiment_id, training):
                    deleted["model_versions"] += discard_run(run)
                    deleted["mlflow_runs"].append(run.info.run_id)
        except Exception as e:
            errors.append(f"MLflow: could not discard the runs of this training: {e}")

    # 3. Dataset uploaded to MinIO
    try:
        deleted["minio_objects"] = delete_training_data(training_id)
    except Exception as e:
        errors.append(f"MinIO: could not delete the dataset: {e}")

    # 4. Does the model keep any version? (checked only if everything above worked)
    others = [t for t in store.list_by_name(experiment_name) if t["training_id"] != training_id]
    has_versions = any(t["status"] == "completed" for t in others)
    if not errors and not has_versions and mlflow_exp is not None:
        try:
            has_versions = has_finished_runs(mlflow_exp.experiment_id, exclude_run_ids=deleted["mlflow_runs"])
        except Exception as e:
            errors.append(f"MLflow: could not check existing versions: {e}")

    if errors:
        # Keep the record so the cleanup can be retried; never delete the experiment in this case.
        store.update(training_id, expected_status=["cancelled"], status_reason=(INCOMPLETE_PREFIX + "; ".join(errors))[:2000])
        print(f"[{training_id}] Cancellation incomplete: {errors}")
        return {"deleted": deleted, "errors": errors, "experiment_deleted": False, "cleanup_complete": False}

    # 5. Remove this training only (by id: a new training of the same model may exist already)
    store.delete(training_id)
    deleted["local_record"] = True

    # 6. Without versions and without another training in progress, the whole model goes away
    experiment_deleted = False
    another_active = any(t["status"] in ACTIVE_STATUSES for t in others)
    if not has_versions and not another_active:
        for t in others:
            store.delete(t["training_id"])
        if mlflow_exp is not None:
            try:
                mlflow.delete_experiment(mlflow_exp.experiment_id)
            except Exception as e:
                errors.append(f"MLflow: could not delete the experiment: {e}")
        experiment_deleted = not errors
        print(f"[{training_id}] Cancelled; experiment '{experiment_name}' deleted (no versions)")
    else:
        print(f"[{training_id}] Cancelled; experiment '{experiment_name}' kept")

    return {"deleted": deleted, "errors": errors, "experiment_deleted": experiment_deleted, "cleanup_complete": True}


def retry_pending_cancellations() -> None:
    """Retries the cleanup of cancelled trainings whose previous cleanup failed."""
    for training in get_store().list_all():
        # Only records whose cleanup already failed once; a cancellation still running
        # in the API request has no reason yet and is left alone.
        if training["status"] != "cancelled" or not (training.get("status_reason") or "").startswith(INCOMPLETE_PREFIX):
            continue
        try:
            cleanup_cancelled_training(training)
        except Exception as e:
            print(f"[{training['training_id']}] Could not retry cancellation cleanup: {e}")
