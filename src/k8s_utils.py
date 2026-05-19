import os
from kubernetes import client, config as k8s_config
from src.database import fake_database
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Extraemos la ruta del .env y la expandimos
raw_kube_path = os.getenv("KUBERNETES_CONFIG_PATH", "~/.kube/config")
kubeconfig_path = os.path.expanduser(raw_kube_path)

# Extraemos el namespace con un valor por defecto seguro
k8s_namespace = os.getenv("KUBERNETES_NAMESPACE", "default")

try:
    # Pass the exact path to the configuration loading function
    k8s_config.load_kube_config(config_file=kubeconfig_path)
    print(f"Kubernetes configuration successfully loaded from: {kubeconfig_path}")
except Exception as e:
    print("FATAL ERROR: Could not load Kubernetes configuration.")
    print(f"Details: {str(e)}")

async def train_model_job(experiment_id: str, model_type: str, config: dict, minio_path: str, experiment_name: str, model_name: str, bucket_name: str = "opentwins-demos"):
    """
    Creates a Kubernetes Job that downloads data from MinIO, trains a model using Ludwig,
    and automatically logs metrics and registers the model in MLflow.
    """
    print(f"[{experiment_id}] Preparing Ludwig + MLflow Job for model {model_type}")
    
    fake_database[experiment_id]["status"] = "launching_job"
    
    env_vars = [
        client.V1EnvVar(name="MINIO_URL", value=os.getenv("MINIO_URL")), 
        client.V1EnvVar(name="MINIO_ACCESS_KEY", value=os.getenv("MINIO_ACCESS_KEY")),
        client.V1EnvVar(name="MINIO_SECRET_KEY", value=os.getenv("MINIO_SECRET_KEY")),
        client.V1EnvVar(name="MLFLOW_TRACKING_URI", value=os.getenv("MLFLOW_TRACKING_URI"))
    ]

    # Convert the config dictionary from the API into a JSON string representation
    import json
    ludwig_config_json = json.dumps(config)

    script_python = f"""
import os
import json
from minio import Minio
from ludwig.api import LudwigModel
import mlflow

print('Starting Advanced MLOps Job in Kubernetes...')

# 1. Establish connections
try:
    minio_client = Minio(
        os.environ.get('MINIO_URL'),
        access_key=os.environ.get('MINIO_ACCESS_KEY'),
        secret_key=os.environ.get('MINIO_SECRET_KEY'),
        secure=True
    )
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment('{experiment_name}')

    # 2. Download dataset from MinIO
    print('Downloading dataset from MinIO...')
    response = minio_client.get_object('{bucket_name}', '{minio_path}')
    
    with open('dataset.csv', 'wb') as f:
        f.write(response.read())
        
except Exception as e:
    print('ERROR IN SETUP PHASE:', str(e))
    exit(1)

# 3. Train and Log
try:
    print('Parsing Ludwig Configuration...')
    ludwig_config = json.loads('''{ludwig_config_json}''')
    ludwig_config['model_name'] = '{model_name}'
    
    print('Initializing Ludwig Model...')
    model = LudwigModel(config=ludwig_config)
    
    print('Starting Training...')
    results = model.train(dataset='dataset.csv')
    
    # Handle both old and new return types for train_stats
    train_stats = results[0] if isinstance(results, tuple) else results
    
    print('Training finished. Logging to MLflow using contrib...')
    try:
        from ludwig.contribs.mlflow.mlflow3 import log_training_run
        
        # Simple wrapper to handle dict stats if needed
        class StatsWrapper:
            def __init__(self, stats):
                self.stats = stats
            def __getattr__(self, name):
                return self.stats.get(name)
        
        stats_to_log = StatsWrapper(train_stats) if isinstance(train_stats, dict) else train_stats
        
        log_training_run(
            model=model,
            train_stats=stats_to_log,
            config=ludwig_config
        )
    except Exception as mlflow_err:
        print('Warning: MLflow logging failed, but training succeeded:', str(mlflow_err))
        # Fallback to direct model logging if log_training_run fails
        try:
            from ludwig.contribs.mlflow.model import log_model
            with mlflow.start_run():
                log_model(model, artifact_path="model", registered_model_name='{model_name}')
        except Exception as e:
            print('Direct logging also failed:', str(e))
        
    print('SUCCESS: Model trained and registration attempt completed.')

except Exception as e:
    print('ERROR DURING TRAINING / MLFLOW LOGGING:', str(e))
    exit(1)
"""

    # 2. Build the bash command
    bash_command = f"pip install minio mlflow -q && cat << 'EOF' > run.py\n{script_python}\nEOF\npython run.py"

    # 3. CRUCIAL CHANGE: Use the official Ludwig Docker image optimized for PyTorch/Python
    container = client.V1Container(
        name="trainer",
        image="ludwigai/ludwig:latest",
        env=env_vars,
        command=["/bin/bash", "-c"],
        args=[bash_command]
    )
    
    pod_spec = client.V1PodSpec(containers=[container], restart_policy="Never")
    
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"experiment_id": experiment_id}),
        spec=pod_spec
    )
    
    job_name = f"train-job-{experiment_id.lower()}"
    job_spec = client.V1JobSpec(template=template, backoff_limit=1)
    
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name),
        spec=job_spec
    )
    
    try:
        batch_v1 = client.BatchV1Api()
        batch_v1.create_namespaced_job(namespace=os.getenv("KUBERNETES_NAMESPACE"), body=job)
        print(f"[{experiment_id}] Ludwig Training Job launched in Kubernetes: {job_name}")
        fake_database[experiment_id]["status"] = "training_in_k8s"
    except Exception as e:
        print(f"[{experiment_id}] Error creating the job in Kubernetes: {str(e)}")
        fake_database[experiment_id]["status"] = "error_launching_k8s_job"
