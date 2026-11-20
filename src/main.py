import asyncio
import contextlib
import json
from contextlib import asynccontextmanager

import mlflow
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from kubernetes.client.rest import ApiException

from src.config import get_settings
from src.database import close_store, init_store
from src.minio_utils import MINIO_ERRORS, ensure_bucket
# Import our modular functions
from src.service import ActiveTrainingExists, DFlowService, NoActiveTraining
from src.training_status import poll_training_status_forever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fails fast (and names the problem) if a required setting is missing,
    # the database is unreachable or MinIO cannot be used.
    settings = get_settings()
    if settings.mlflow_enabled:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    init_store(settings)
    ensure_bucket()

    poller = asyncio.create_task(poll_training_status_forever())
    try:
        yield
    finally:
        poller.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poller
        close_store()


app = FastAPI(
    title="dflow API",
    description="API to orchestrate ML model deployments and training",
    version="1.0.0",
    lifespan=lifespan,
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = DFlowService()


def mlflow_enabled() -> bool:
    return get_settings().mlflow_enabled


@app.get("/")
def home():
    """Root endpoint to verify the API is running."""
    return {"message": "Welcome to the dflow API. Go to /docs to view the interface."}

@app.get("/api/mlops/experiments")
def get_experiments(page: int = Query(0, ge=0), pageSize: int = Query(20, ge=1, le=1000), search: str = ''):
    """
    Returns one entry per experiment (model), merging local trainings and MLflow history.
    `status` is one of: queued, launching_job, training, completed, failed
    (`status_reason` explains failures).
    """

    return service.get_all_experiments(mlflow_enabled(), page, pageSize, search)

@app.get("/api/mlops/experiments/{experiment_name}")
async def get_experiment(experiment_name: str):
    try:
        return await service.get_experiment(experiment_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"There is no experiment with name {str(experiment_name)}")

@app.get("/api/mlops/experiments/{experiment_name}/logs")
async def get_training_logs(experiment_name: str, tail_lines: int = Query(500, ge=1, le=5000)):
    """Returns the last `tail_lines` log lines of the training pod of an experiment."""
    try:
        return await service.get_training_logs(experiment_name, tail_lines)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"There is no experiment with name {str(experiment_name)}")
    except ApiException as e:
        raise HTTPException(status_code=502, detail=f"Kubernetes error while reading training logs: {e.status} {e.reason}")

@app.post("/api/mlops/experiments")
async def create_experiment(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    model_type: str = Form(...),
    config_json: str = Form(...),
    dependencies: str = Form("[]"),
    experiment_name: str = Form("dflow-default-experiment"),
    model_name: str = Form("ludwig_model")
):
    """
    Receives the data file, model type, configuration, and dependencies to create a new experiment.
    Uploads the data to MinIO and triggers a Kubernetes Job in the background.
    Returns 409 if the experiment already has a training in progress.
    """
    # 1. Parse and validate JSON inputs
    try:
        config_dict = json.loads(config_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="config_json is not a valid JSON string")

    try:
        dependencies_list = json.loads(dependencies)
    except json.JSONDecodeError:
        # Fallback for comma-separated strings
        if dependencies.strip():
            dependencies_list = [dep.strip() for dep in dependencies.split(",")]
        else:
            dependencies_list = []

    try:
        experiment = await service.create_experiment(background_tasks, csv_file, model_type, experiment_name, model_name, config_dict, dependencies_list)
    except ActiveTrainingExists:
        raise HTTPException(
            status_code=409,
            detail=f"Experiment '{experiment_name}' already has a training in progress. Cancel it or wait until it finishes.",
        )
    experiment_id = experiment["internal_id"]
    return {
        "message": f"Experiment {experiment_id} successfully created",
        "experiment_id": experiment_id,
        "details": experiment
    }


@app.delete("/api/mlops/experiments/{experiment_name}")
def delete_experiment(experiment_name: str):
    """Deletes an experiment by its NAME from the local DB, MLflow and MinIO (training CSVs of all its trainings).

    If a training is in progress, its Kubernetes Job is stopped first. If that fails, or if the
    MinIO data cannot be deleted, nothing else is deleted and 502 is returned so the request
    can be retried.
    """
    try:
        deleted = service.delete_experiment(experiment_name, mlflowb=mlflow_enabled())
    except ApiException as e:
        raise HTTPException(status_code=502, detail=f"Kubernetes error while deleting the training job: {e.status} {e.reason}")
    except MINIO_ERRORS as e:
        raise HTTPException(status_code=502, detail=f"MinIO error while deleting training data: {e}")

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found in local DB or MLflow")

    return {"message": f"Experiment '{experiment_name}' successfully deleted"}

@app.post("/api/mlops/experiments/{experiment_name}/cancel")
def cancel_training(experiment_name: str):
    """
    Stops the training in progress (queued, launching_job or training) of an experiment and removes it:
    deletes the Kubernetes Job and its pods, kills and deletes its MLflow run and deletes its dataset
    from MinIO. If the model has no previous versions the whole experiment is deleted; otherwise only
    this training is removed and the existing versions are kept.
    If a step fails, `cleanup_complete` is false: the training stays as 'cancelled' and the cleanup is
    retried automatically (calling this endpoint again also retries it).
    """
    try:
        return service.cancel_training(experiment_name)
    except NoActiveTraining:
        raise HTTPException(status_code=404, detail=f"No training in progress for experiment '{experiment_name}'")

@app.get("/api/mlops/experiments/{experiment_name}/runs")
def get_runs(experiment_name: str, page: int = Query(0, ge=0), pageSize: int = Query(20, ge=1, le=1000), search: str = ''):
    """
    Returns the list of runs for a given experiment name from MLflow.
    Includes metrics, parameters, and tags for each run.
    """
    if not mlflow_enabled():
        raise HTTPException(status_code=500, detail="MLflow tracking URI is not configured")

    try:
        result = service.get_runs(experiment_name, page, pageSize, search)
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail=f"There is no experiment with name {str(experiment_name)}")
    except Exception as e:
        print(f"Error searching runs in MLflow for '{experiment_name}': {str(e)}")
        # If it's a known error from mlflow or pandas, we catch it
        raise HTTPException(status_code=500, detail=f"Error searching runs in MLflow: {str(e)}")
