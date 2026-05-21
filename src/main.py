from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import os
import mlflow

# Import our modular functions
# Importamos nuestras funciones modulares indicando la carpeta src
from src.database import fake_database
from src.minio_utils import upload_csv_to_minio
from src.k8s_utils import train_model_job

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

@app.get("/")
def home():
    """Root endpoint to verify the API is running."""
    return {"message": "Welcome to the dflow API. Go to /docs to view the interface."}

@app.get("/api/mlops/experiments")
def get_experiments():
    """
    Returns experiments from both local tracking (K8s Jobs in progress) 
    and MLflow history (completed/archived), indexed by their name.
    """
    # Use name as the key for the merged dict to simplify frontend interaction
    merged_experiments = {}
    
    # 1. Start with the local database (currently running or pending jobs)
    for eid, data in fake_database.items():
        name = data.get("experiment_name", f"local-{eid}")
        merged_experiments[name] = {
            "internal_id": eid,
            **data
        }

    # 2. Query MLflow for persistent history
    if mlflow_uri:
        try:
            # Fetch all active experiments from the MLflow server
            mlflow_exps = mlflow.search_experiments()
            
            for exp in mlflow_exps:
                # Skip the default empty experiment
                if exp.name == "Default":
                    continue
                    
                # If name already exists in merged (from local), we keep the local version
                # as it contains more real-time status (like "training_in_k8s")
                if exp.name not in merged_experiments:
                    merged_experiments[exp.name] = {
                        "experiment_name": exp.name,
                        "status": "completed (mlflow)",
                        "model": "historical_or_external",
                        "mlflow_id": exp.experiment_id,
                        "artifact_location": exp.artifact_location
                    }
        except Exception as e:
            print(f"Error reading from MLflow: {str(e)}")

    return {"experiments": merged_experiments}

@app.post("/api/mlops/experiment")
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

    file_content = await csv_file.read()
    experiment_id = str(uuid.uuid4())[:8]
    
    # 2. Delegate the file upload to MinIO
    minio_path = upload_csv_to_minio(file_content, experiment_id, csv_file.filename)

    # 3. Save initial state in the mock database
    fake_database[experiment_id] = {
        "status": "pending_training",
        "model": model_type,
        "config": config_dict,
        "dependencies": dependencies_list,
        "csv_filename": csv_file.filename,
        "experiment_name": experiment_name,
        "model_name": model_name
    }
    
    # 4. Delegate the execution to Kubernetes as a background task
    background_tasks.add_task(
        train_model_job,
        experiment_id=experiment_id,
        model_type=model_type,
        config=config_dict,
        minio_path=minio_path,
        experiment_name=experiment_name,
        model_name=model_name
    )
    
    return {
        "message": f"Experiment {experiment_id} successfully created", 
        "experiment_id": experiment_id,
        "details": fake_database[experiment_id]
    }

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

@app.delete("/api/mlops/experiment/{experiment_name}")
def delete_experiment(experiment_name: str):
    """Deletes an experiment by its NAME from both the local DB and MLflow."""
    found = False

    # 1. Search and delete from local database
    keys_to_delete = [k for k, v in fake_database.items() if v.get("experiment_name") == experiment_name]
    for k in keys_to_delete:
        del fake_database[k]
        found = True
        print(f"Local DB: Experiment '{experiment_name}' (ID: {k}) deleted.")

    # 2. Search and delete from MLflow
    if mlflow_uri:
        try:
            # Buscar el experimento en MLflow por su nombre
            mlflow_exp = mlflow.get_experiment_by_name(experiment_name)

            # Si existe y no está ya borrado, lo eliminamos
            if mlflow_exp and mlflow_exp.lifecycle_stage != "deleted":
                mlflow.delete_experiment(mlflow_exp.experiment_id)
                found = True
                print(f"MLflow: Experiment '{experiment_name}' deleted.")
        except Exception as e:
            print(f"Error deleting from MLflow: {str(e)}")

    if not found:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found in local DB or MLflow")

    return {"message": f"Experiment '{experiment_name}' successfully deleted"}