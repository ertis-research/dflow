"""Builds and launches the Kubernetes Job that trains a model."""
import asyncio
import json
from pathlib import Path

from kubernetes import client

from src.config import get_settings
from src.db.store import get_store
from src.integrations.k8s import (
    CONTAINER_NAME,
    TRAINING_ID_LABEL,
    batch_api,
    delete_training_job,
    job_name_for,
    truncate_reason,
)

# Source of the script the pod runs (see trainer_script.py).
TRAINING_SCRIPT = (Path(__file__).parent / "trainer_script.py").read_text(encoding="utf-8")


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
            status_reason=truncate_reason(f"Could not create the Kubernetes job: {e}"),
        )
        return

    print(f"[{training_id}] Ludwig Training Job launched in Kubernetes: {job_name}")
    if not store.update(training_id, expected_status=["launching_job"], status="training", job_name=job_name):
        # Cancelled while the Job was being created: remove the Job we just made.
        print(f"[{training_id}] Training cancelled during launch; deleting job {job_name}")
        await asyncio.to_thread(delete_training_job, job_name)
