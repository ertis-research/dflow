from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import mlflow

# Import our modular functions
# Importamos nuestras funciones modulares indicando la carpeta src
from src.service import DFlowService

app = FastAPI(
    title="dflow API",
    description="API to orchestrate ML model deployments and training",
    version="1.0.0"
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

@app.get("/")
def home():
    """Root endpoint to verify the API is running."""
    return {"message": "Welcome to the dflow API. Go to /docs to view the interface."}

@app.get("/api/mlops/experiments")
def get_experiments(page: int = 0, pageSize: int = 20, search: str = ''):
    """
    Returns experiments from both local tracking (K8s Jobs in progress) 
    and MLflow history (completed/archived), indexed by their name.
    """

    return service.get_all_experiments(mlflow_uri is not None and len(mlflow_uri)>0, page, pageSize, search)

@app.get("/api/mlops/experiments/{experiment_name}")
async def get_experiment(experiment_name: str):
    try:
        return await service.get_experiment(experiment_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"There is no experiment with name {str(experiment_name)}")

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


    experiment = await service.create_experiment(background_tasks, csv_file, model_type, experiment_name, model_name, config_dict, dependencies_list)
    experiment_id = experiment["internal_id"]
    return {
        "message": f"Experiment {experiment_id} successfully created", 
        "experiment_id": experiment_id,
        "details": experiment
    }
    

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

@app.delete("/api/mlops/experiments/{experiment_name}")
def delete_experiment(experiment_name: str):
    """Deletes an experiment by its NAME from both the local DB and MLflow."""
    deleted = service.delete_experiment(experiment_name, mlflowb=mlflow_uri is not None and len(mlflow_uri)>0)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found in local DB or MLflow")

    return {"message": f"Experiment '{experiment_name}' successfully deleted"}

@app.get("/api/mlops/experiments/{experiment_name}/runs")
def get_runs(experiment_name: str, page: int = 0, pageSize: int = 20, search: str = ''):
    """
    Returns the list of runs for a given experiment name from MLflow.
    Includes metrics, parameters, and tags for each run.
    """
    if not mlflow_uri:
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