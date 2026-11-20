"""Training script run INSIDE the training pod (not imported by dflow).

dflow writes this file into the Job container and runs it with python. Every per-training
value arrives through environment variables, so names or configs with quotes cannot break
(or inject into) the code."""
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
