import asyncio
import json
import threading
from typing import Optional, Tuple

from kubernetes import client, config as k8s_config
from kubernetes.client.rest import ApiException

from src.config import get_settings
from src.database import get_store

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


def _truncate(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_REASON_LENGTH else text[: MAX_REASON_LENGTH - 3] + "..."


# The script is a constant: every per-training value reaches it through environment variables,
# so names or configs with quotes cannot break (or inject into) the generated code.
TRAINING_SCRIPT = r"""
import json
import os
import sys
import time

from minio import Minio
from ludwig.api import LudwigModel
import mlflow
from mlflow.tracking import MlflowClient

training_id = os.environ['DFLOW_TRAINING_ID']
experiment_name = os.environ['DFLOW_EXPERIMENT_NAME']
model_name = os.environ['DFLOW_MODEL_NAME']
bucket_name = os.environ['DFLOW_BUCKET_NAME']
minio_path = os.environ['DFLOW_MINIO_PATH']
started_at_ms = int(time.time() * 1000)


def fail(phase, err):
    # dflow reads the last 'ERROR' line of these logs as the failure reason.
    print(f'ERROR {phase}: {err}', flush=True)
    sys.exit(1)


print('Starting Advanced MLOps Job in Kubernetes...', flush=True)

# 1. Establish connections and download the dataset
try:
    minio_client = Minio(
        os.environ['MINIO_URL'],
        access_key=os.environ['MINIO_ACCESS_KEY'],
        secret_key=os.environ['MINIO_SECRET_KEY'],
        secure=os.environ.get('MINIO_SECURE', 'true').lower() == 'true',
    )
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(experiment_name)

    print('Downloading dataset from MinIO...', flush=True)
    response = minio_client.get_object(bucket_name, minio_path)
    with open('dataset.csv', 'wb') as f:
        f.write(response.read())
except Exception as e:
    fail('IN SETUP PHASE', e)

# 2. Train
try:
    print('Parsing Ludwig Configuration...', flush=True)
    ludwig_config = json.loads(os.environ['DFLOW_LUDWIG_CONFIG'])
    ludwig_config['model_name'] = model_name

    print('Initializing Ludwig Model...', flush=True)
    model = LudwigModel(config=ludwig_config)

    print('Starting Training...', flush=True)
    results = model.train(dataset='dataset.csv')
    # Handle both old and new return types for train_stats
    train_stats = results[0] if isinstance(results, tuple) else results
except Exception as e:
    fail('DURING TRAINING', e)

# 3. Log and register in MLflow
print('Training finished. Logging to MLflow...', flush=True)
run_id = None
try:
    from ludwig.contribs.mlflow.mlflow3 import log_training_run

    class StatsWrapper:
        def __init__(self, stats):
            self.stats = stats
        def __getattr__(self, name):
            return self.stats.get(name)

    stats_to_log = StatsWrapper(train_stats) if isinstance(train_stats, dict) else train_stats
    log_training_run(model=model, train_stats=stats_to_log, config=ludwig_config)
    last_run = mlflow.last_active_run()
    run_id = last_run.info.run_id if last_run else None
except Exception as mlflow_err:
    print('Warning: MLflow contrib logging failed, trying direct model logging:', str(mlflow_err), flush=True)
    try:
        from ludwig.contribs.mlflow.model import log_model
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            log_model(model, artifact_path='model', registered_model_name=model_name)
    except Exception as e:
        fail('DURING MLFLOW LOGGING', e)

if run_id is None:
    # Fallback: the newest run of this experiment created by this pod.
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f'attributes.start_time >= {started_at_ms}',
        order_by=['attributes.start_time DESC'],
        max_results=1,
        output_format='list',
    )
    run_id = runs[0].info.run_id if runs else None

if run_id is None:
    fail('DURING MLFLOW LOGGING', 'the model was trained but no MLflow run was created')

try:
    MlflowClient().set_tag(run_id, 'dflow.training_id', training_id)
except Exception as e:
    print('Warning: could not tag the MLflow run:', str(e), flush=True)

print('SUCCESS: Model trained and registered in MLflow (run ' + run_id + ').', flush=True)
"""


def build_training_job(training_id: str, config: dict, minio_path: str, experiment_name: str, model_name: str) -> client.V1Job:
    settings = get_settings()
    env = {
        "MINIO_URL": settings.minio_url,
        "MINIO_ACCESS_KEY": settings.minio_access_key,
        "MINIO_SECRET_KEY": settings.minio_secret_key,
        "MINIO_SECURE": "true" if settings.minio_secure else "false",
        "MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri or "",
        "DFLOW_TRAINING_ID": training_id,
        "DFLOW_EXPERIMENT_NAME": experiment_name,
        "DFLOW_MODEL_NAME": model_name,
        "DFLOW_BUCKET_NAME": settings.bucket_name,
        "DFLOW_MINIO_PATH": minio_path,
        "DFLOW_LUDWIG_CONFIG": json.dumps(config),
    }
    bash_command = f"pip install minio mlflow -q && cat << 'EOF' > run.py\n{TRAINING_SCRIPT}\nEOF\npython -u run.py"

    container = client.V1Container(
        name=CONTAINER_NAME,
        image=settings.training_image,
        env=[client.V1EnvVar(name=k, value=v) for k, v in env.items()],
        command=["/bin/bash", "-c"],
        args=[bash_command],
    )
    labels = {TRAINING_ID_LABEL: training_id}
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=labels),
        spec=client.V1PodSpec(containers=[container], restart_policy="Never"),
    )
    job_spec = client.V1JobSpec(
        template=template,
        backoff_limit=settings.training_job_backoff_limit,
        ttl_seconds_after_finished=settings.training_job_ttl_seconds,
    )
    return client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name_for(training_id), labels=labels),
        spec=job_spec,
    )


async def train_model_job(training_id: str, model_type: str, config: dict, minio_path: str, experiment_name: str, model_name: str) -> None:
    """
    Creates a Kubernetes Job that downloads data from MinIO, trains a model using Ludwig,
    and logs metrics and registers the model in MLflow.
    Status transitions: queued -> launching_job -> training (or failed).
    """
    store = get_store()
    # Compare-and-set: if the training was cancelled while queued, do nothing.
    if not store.update(training_id, expected_status=["queued"], status="launching_job"):
        print(f"[{training_id}] Not launching: the training is no longer queued")
        return

    print(f"[{training_id}] Preparing Ludwig + MLflow Job for model {model_type}")
    job_name = job_name_for(training_id)
    try:
        job = build_training_job(training_id, config, minio_path, experiment_name, model_name)
        await asyncio.to_thread(batch_api().create_namespaced_job, get_settings().k8s_namespace, job)
    except Exception as e:
        print(f"[{training_id}] Error creating the job in Kubernetes: {str(e)}")
        store.update(
            training_id,
            expected_status=["launching_job"],
            status="failed",
            status_reason=_truncate(f"Could not create the Kubernetes job: {e}"),
        )
        return

    print(f"[{training_id}] Ludwig Training Job launched in Kubernetes: {job_name}")
    if not store.update(training_id, expected_status=["launching_job"], status="training", job_name=job_name):
        # Cancelled while the Job was being created: remove the Job we just made.
        print(f"[{training_id}] Training cancelled during launch; deleting job {job_name}")
        await asyncio.to_thread(delete_training_job, job_name)


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


def _job_pods(job_name: str) -> list:
    pods = core_api().list_namespaced_pod(get_settings().k8s_namespace, label_selector=f"job-name={job_name}").items
    # Newest first: with retries, the last attempt is the most relevant one.
    return sorted(pods, key=lambda p: p.metadata.creation_timestamp, reverse=True)


def get_pod_failure_reason(job_name: str) -> Optional[str]:
    """Last 'ERROR' line of the training pod logs, or how its container terminated.
    Returns None if nothing can be read (pods already gone, or no permission)."""
    try:
        pods = _job_pods(job_name)
    except ApiException as e:
        print(f"Cannot list pods of {job_name}: {e.status} {e.reason}")
        return None

    namespace = get_settings().k8s_namespace
    for pod in pods:
        try:
            logs = core_api().read_namespaced_pod_log(pod.metadata.name, namespace, container=CONTAINER_NAME, tail_lines=50)
            error_lines = [line for line in (logs or "").splitlines() if "ERROR" in line]
            if error_lines:
                return _truncate(error_lines[-1])
        except ApiException as e:
            print(f"Cannot read logs of {pod.metadata.name}: {e.status} {e.reason}")

        for cs in (pod.status.container_statuses or []) if pod.status else []:
            terminated = cs.state.terminated if cs.state else None
            if terminated and terminated.exit_code:
                parts = [p for p in (terminated.reason, terminated.message) if p]
                detail = ": ".join(parts) if parts else "container failed"
                return _truncate(f"{detail} (exit code {terminated.exit_code})")
    return None


def get_pod_waiting_reason(job_name: str) -> Optional[str]:
    """Why a training pod is stuck (image pull errors, unschedulable...), or None if it is fine."""
    try:
        pods = _job_pods(job_name)
    except ApiException:
        return None
    for pod in pods:
        status = pod.status
        if not status:
            continue
        for cs in status.container_statuses or []:
            waiting = cs.state.waiting if cs.state else None
            if waiting and waiting.reason in STUCK_WAITING_REASONS:
                return _truncate(f"{waiting.reason}: {waiting.message}" if waiting.message else waiting.reason)
        for condition in status.conditions or []:
            if condition.type == "PodScheduled" and condition.status == "False" and condition.reason == "Unschedulable":
                return _truncate(f"Unschedulable: {condition.message}" if condition.message else "Unschedulable")
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
