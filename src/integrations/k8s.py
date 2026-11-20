import threading
from typing import Optional, Tuple

from kubernetes import client, config as k8s_config
from kubernetes.client.rest import ApiException

from src.config import get_settings

TRAINING_ID_LABEL = "dflow/training-id"
CONTAINER_NAME = "trainer"
MAX_REASON_LENGTH = 500

# Pod waiting reasons that mean the training is stuck, shown to the user as status_reason.
STUCK_WAITING_REASONS = {
    "ImagePullBackOff",
    "ErrImagePull",
    "InvalidImageName",
    "CreateContainerConfigError",
    "CreateContainerError",
    "CrashLoopBackOff",
}

_config_lock = threading.Lock()
_config_loaded = False


def _load_k8s_config() -> None:
    """In-cluster config when running inside Kubernetes, kubeconfig otherwise. Loaded once."""
    global _config_loaded
    with _config_lock:
        if _config_loaded:
            return
        try:
            k8s_config.load_incluster_config()
            print("Kubernetes configuration loaded from the cluster service account")
        except k8s_config.ConfigException:
            path = get_settings().kube_config_path
            k8s_config.load_kube_config(config_file=path)
            print(f"Kubernetes configuration loaded from: {path}")
        _config_loaded = True


def batch_api() -> client.BatchV1Api:
    _load_k8s_config()
    return client.BatchV1Api()


def core_api() -> client.CoreV1Api:
    _load_k8s_config()
    return client.CoreV1Api()


def job_name_for(training_id: str) -> str:
    return f"dflow-train-job-{training_id.lower()}"


def truncate_reason(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_REASON_LENGTH else text[: MAX_REASON_LENGTH - 3] + "..."


# ---- Job inspection ---------------------------------------------------------


def read_training_job(job_name: str) -> Optional[client.V1Job]:
    """The Job, or None if it does not exist (deleted, or removed by its TTL)."""
    try:
        return batch_api().read_namespaced_job(job_name, get_settings().k8s_namespace)
    except ApiException as e:
        if e.status == 404:
            return None
        raise


def job_state(job: client.V1Job) -> Tuple[str, Optional[str]]:
    """Returns ('succeeded' | 'failed' | 'active', reason)."""
    status = job.status
    for condition in (status.conditions or []) if status else []:
        if condition.status != "True":
            continue
        if condition.type == "Complete":
            return "succeeded", None
        if condition.type == "Failed":
            parts = [p for p in (condition.reason, condition.message) if p]
            return "failed", ": ".join(parts) if parts else None
    if status and status.succeeded:
        return "succeeded", None
    return "active", None


def job_pods(job_name: str) -> list:
    pods = core_api().list_namespaced_pod(get_settings().k8s_namespace, label_selector=f"job-name={job_name}").items
    # Newest first: with retries, the last attempt is the most relevant one.
    return sorted(pods, key=lambda p: p.metadata.creation_timestamp, reverse=True)


def get_pod_failure_reason(job_name: str) -> Optional[str]:
    """Last 'ERROR' line of the training pod logs, or how its container terminated.
    Returns None if nothing can be read (pods already gone, or no permission)."""
    try:
        pods = job_pods(job_name)
    except ApiException as e:
        print(f"Cannot list pods of {job_name}: {e.status} {e.reason}")
        return None

    namespace = get_settings().k8s_namespace
    for pod in pods:
        try:
            logs = core_api().read_namespaced_pod_log(pod.metadata.name, namespace, container=CONTAINER_NAME, tail_lines=50)
            error_lines = [line for line in (logs or "").splitlines() if "ERROR" in line]
            if error_lines:
                return truncate_reason(error_lines[-1])
        except ApiException as e:
            print(f"Cannot read logs of {pod.metadata.name}: {e.status} {e.reason}")

        for cs in (pod.status.container_statuses or []) if pod.status else []:
            terminated = cs.state.terminated if cs.state else None
            if terminated and terminated.exit_code:
                parts = [p for p in (terminated.reason, terminated.message) if p]
                detail = ": ".join(parts) if parts else "container failed"
                return truncate_reason(f"{detail} (exit code {terminated.exit_code})")
    return None


def get_pod_waiting_reason(job_name: str) -> Optional[str]:
    """Why a training pod is stuck (image pull errors, unschedulable...), or None if it is fine."""
    try:
        pods = job_pods(job_name)
    except ApiException:
        return None
    for pod in pods:
        status = pod.status
        if not status:
            continue
        for cs in status.container_statuses or []:
            waiting = cs.state.waiting if cs.state else None
            if waiting and waiting.reason in STUCK_WAITING_REASONS:
                return truncate_reason(f"{waiting.reason}: {waiting.message}" if waiting.message else waiting.reason)
        for condition in status.conditions or []:
            if condition.type == "PodScheduled" and condition.status == "False" and condition.reason == "Unschedulable":
                return truncate_reason(f"Unschedulable: {condition.message}" if condition.message else "Unschedulable")
    return None


def delete_training_job(job_name: str) -> bool:
    """Deletes the Job; Kubernetes garbage-collects its pods. Returns False if it did not exist."""
    try:
        batch_api().delete_namespaced_job(job_name, get_settings().k8s_namespace, propagation_policy="Background")
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        raise
