"""Operations on the training of an experiment: reading its pod logs and cancelling it."""
import asyncio
from typing import Any, Dict, Optional

from kubernetes.client.rest import ApiException

from src.config import get_settings
from src.db.store import get_store
from src.integrations.k8s import CONTAINER_NAME, core_api, get_pod_waiting_reason, job_name_for, job_pods
from src.integrations.mlflow_tracking import get_active_experiment
from src.schemas import ACTIVE_STATUSES, TrainingLogs
from src.services.experiments import resolve_status, summarize_runs
from src.training.cancellation import cleanup_cancelled_training
from src.training.status import reconcile_training


class NoActiveTraining(Exception):
    """Raised when cancelling an experiment that has no training in progress."""


async def get_training_logs(experiment_name: str, tail_lines: int) -> TrainingLogs:
    """Logs of the training pod of an experiment (the active training, or else the latest one).
    Raises KeyError if the experiment does not exist. Non-404 Kubernetes errors are re-raised."""
    store = get_store()
    active = store.active(experiment_name)
    training = active[0] if active else store.latest(experiment_name)

    def result(status: str, available: bool, reason: Optional[str], logs: str = "",
               training_id: Optional[str] = None, job_name: Optional[str] = None,
               pod_name: Optional[str] = None, pod_phase: Optional[str] = None) -> TrainingLogs:
        return TrainingLogs(
            experiment_name=experiment_name,
            training_id=training_id,
            job_name=job_name,
            pod_name=pod_name,
            status=status,
            pod_phase=pod_phase,
            logs=logs,
            available=available,
            reason=reason,
        )

    if training is None:
        mlflow_exp = get_active_experiment(experiment_name) if get_settings().mlflow_enabled else None
        if mlflow_exp is None:
            raise KeyError()
        summary = summarize_runs([mlflow_exp.experiment_id]).get(mlflow_exp.experiment_id)
        status, _ = resolve_status(None, summary)
        return result(status, False, "This model was not trained by dflow, or its training is no longer kept.")

    training_id = training["training_id"]
    job_name = training.get("job_name") or job_name_for(training_id)
    status = training["status"]
    ids = {"training_id": training_id, "job_name": job_name}

    if status in ("queued", "launching_job"):
        return result(status, False, "The training has not started yet.", **ids)

    no_pod = "The training pod no longer exists; its logs are not kept."
    try:
        pods = await asyncio.to_thread(job_pods, job_name)
    except ApiException as e:
        if e.status != 404:
            raise
        pods = []
    if not pods:
        return result(status, False, no_pod, **ids)

    pod = pods[0]
    pod_name = pod.metadata.name
    pod_phase = pod.status.phase if pod.status else None
    ids.update(pod_name=pod_name, pod_phase=pod_phase)

    def read_logs() -> str:
        return core_api().read_namespaced_pod_log(
            pod_name,
            get_settings().k8s_namespace,
            container=CONTAINER_NAME,
            tail_lines=tail_lines,
            timestamps=False,
        )

    try:
        logs = await asyncio.to_thread(read_logs)
    except ApiException as e:
        if e.status == 400:
            waiting = await asyncio.to_thread(get_pod_waiting_reason, job_name)
            reason = "The training pod is starting." + (f" ({waiting})" if waiting else "")
            return result(status, False, reason, **ids)
        if e.status == 404:
            return result(status, False, no_pod, **ids)
        raise

    return result(status, True, None, logs or "", **ids)


def cancel_training(experiment_name: str) -> Dict[str, Any]:
    """Stops the training in progress of an experiment and removes it.

    Deletes the Kubernetes Job (and its pods), kills and deletes the MLflow run of this
    training and its uploaded dataset. If the model has no other versions, the whole
    experiment is deleted; otherwise only this training is removed and the existing
    versions are left untouched. Raises NoActiveTraining if nothing is in progress.
    """
    store = get_store()

    active = store.active(experiment_name)
    if active:
        training_id = active[0]["training_id"]

        # Refresh first: the training may have just finished.
        try:
            reconcile_training(training_id)
        except Exception as e:
            print(f"[{training_id}] Could not refresh status before cancelling: {e}")

        # Compare-and-set to 'cancelled': from now on the launcher and the poller leave it alone.
        if not store.update(training_id, expected_status=ACTIVE_STATUSES, status="cancelled"):
            raise NoActiveTraining(experiment_name)
    else:
        # A previous cancellation whose cleanup failed: retry it.
        pending = [t for t in store.list_by_name(experiment_name) if t["status"] == "cancelled"]
        if not pending:
            raise NoActiveTraining(experiment_name)
        training_id = pending[0]["training_id"]

    training = store.get(training_id)
    if training is None:
        raise NoActiveTraining(experiment_name)

    result = cleanup_cancelled_training(training)
    if not result["cleanup_complete"]:
        outcome = "cleanup incomplete, it will be retried automatically"
    elif result["experiment_deleted"]:
        outcome = "experiment deleted because it had no versions"
    else:
        outcome = "the existing model versions were kept"

    return {
        "message": f"Training '{training_id}' of '{experiment_name}' cancelled; {outcome}",
        "experiment_name": experiment_name,
        "training_id": training_id,
        "status": "cancelled",
        "experiment_deleted": result["experiment_deleted"],
        "cleanup_complete": result["cleanup_complete"],
        "deleted": result["deleted"],
        "errors": result["errors"],
    }
