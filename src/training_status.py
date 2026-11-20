"""
Keeps the status of each training in sync with Kubernetes and MLflow.

The training Job is removed by Kubernetes shortly after it finishes (TTL), so a background
poller inspects active trainings every few seconds and records the final status (and the failure
reason, read from the pod logs) while the Job and its pods still exist.
"""
import asyncio
from typing import Any, Dict, Optional

from src.cancellation import retry_pending_cancellations
from src.config import get_settings
from src.database import get_store, now_ms
from src.k8s_utils import (
    get_pod_failure_reason,
    get_pod_waiting_reason,
    job_state,
    read_training_job,
)
from src.mlflow_utils import find_finished_training_run
from src.schemas import ACTIVE_STATUSES

REASON_JOB_GONE = "The training job no longer exists"
REASON_NOT_REGISTERED = "Training finished but the model was not registered in MLflow"
REASON_NEVER_LAUNCHED = "The training job was never launched"
REASON_JOB_FAILED = "The training job failed"


def _finish(training: Dict[str, Any], status: str, reason: Optional[str] = None, run_id: Optional[str] = None) -> None:
    fields: Dict[str, Any] = {"status": status, "status_reason": reason}
    if run_id:
        fields["mlflow_run_id"] = run_id
    # Compare-and-set against the status we read, so a concurrent cancel is never overwritten.
    if get_store().update(training["training_id"], expected_status=[training["status"]], **fields):
        print(f"[{training['training_id']}] {training['status']} -> {status}" + (f" ({reason})" if reason else ""))


def _finished_run_id(training: Dict[str, Any]) -> Optional[str]:
    if not get_settings().mlflow_enabled:
        return None
    run = find_finished_training_run(training)
    return run.info.run_id if run else None


def reconcile_training(training_id: str) -> Optional[Dict[str, Any]]:
    """Updates one training from Kubernetes/MLflow and returns it (None if it does not exist).
    Only trainings in an active status are inspected; errors talking to Kubernetes or MLflow
    propagate and leave the stored status untouched."""
    store = get_store()
    training = store.get(training_id)
    if training is None or training["status"] not in ACTIVE_STATUSES:
        return training

    settings = get_settings()
    status = training["status"]

    if status in ("queued", "launching_job"):
        if now_ms() - int(training["created_at"]) > settings.launch_timeout_seconds * 1000:
            _finish(training, "failed", REASON_NEVER_LAUNCHED)
        return store.get(training_id)

    # status == "training"
    job_name = training["job_name"]
    job = read_training_job(job_name)

    if job is None:
        # Deleted by hand or by its TTL. The MLflow run, if any, tells how it ended.
        run_id = _finished_run_id(training)
        if run_id:
            _finish(training, "completed", run_id=run_id)
        else:
            _finish(training, "failed", REASON_JOB_GONE)
        return store.get(training_id)

    state, job_reason = job_state(job)
    if state == "succeeded":
        if not settings.mlflow_enabled:
            _finish(training, "completed")
        else:
            run_id = _finished_run_id(training)
            if run_id:
                _finish(training, "completed", run_id=run_id)
            else:
                _finish(training, "failed", REASON_NOT_REGISTERED)
    elif state == "failed":
        reason = get_pod_failure_reason(job_name) or job_reason or REASON_JOB_FAILED
        _finish(training, "failed", reason)
    else:
        waiting = get_pod_waiting_reason(job_name)
        if waiting != training.get("status_reason"):
            store.update(training_id, expected_status=["training"], status_reason=waiting)

    return store.get(training_id)


def reconcile_active_trainings() -> None:
    for training in get_store().active():
        try:
            reconcile_training(training["training_id"])
        except Exception as e:
            print(f"[{training['training_id']}] Could not refresh training status: {e}")
    # Cancellations whose cleanup failed (MinIO/MLflow/Kubernetes down) are retried here.
    retry_pending_cancellations()


async def poll_training_status_forever() -> None:
    interval = get_settings().status_poll_interval_seconds
    print(f"Training status poller started (every {interval}s)")
    while True:
        try:
            await asyncio.to_thread(reconcile_active_trainings)
        except Exception as e:
            print(f"Training status poller error: {e}")
        await asyncio.sleep(interval)
