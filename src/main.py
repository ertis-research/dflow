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

@app.get("/api/mlops/runs/{experiment_name}")
def get_runs(experiment_name: str):
    """
    Returns the list of runs for a given experiment name from MLflow.
    Includes metrics, parameters, and tags for each run.
    """
    if not mlflow_uri:
        raise HTTPException(status_code=500, detail="MLflow tracking URI is not configured")

    try:
        # 1. Find the experiment by name in MLflow
        mlflow_exp = mlflow.get_experiment_by_name(experiment_name)
        if not mlflow_exp:
            # If the experiment doesn't exist in MLflow yet, return an empty list
            return {"runs": []}

        # 2. Search runs for this experiment, ordered by start_time descending (latest first)
        # Note: MLflow search_runs can return a lot of columns, we'll clean them up
        runs_df = mlflow.search_runs(
            experiment_ids=[mlflow_exp.experiment_id],
            order_by=["attribute.start_time DESC"]
        )
        
        # 3. Process the DataFrame to ensure JSON serializable types and consistent keys
        runs = []
        for _, row in runs_df.iterrows():
            run_dict = row.to_dict()
            
            # Ensure we have consistent base keys for the frontend
            clean_run = {
                "run_id": str(row.get("run_id", "")),
                "status": str(row.get("status", "UNKNOWN")),
                # Convert start_time to ISO format string if it's a datetime
                "start_time": row["start_time"].isoformat() if hasattr(row["start_time"], "isoformat") else str(row["start_time"]),
            }
            
            # Add all other metrics, params, and tags
            for key, value in run_dict.items():
                if key not in ["run_id", "status", "start_time"]:
                    # Handle NaN/None for JSON
                    if value != value or value is None: # value != value is a check for NaN
                        clean_run[key] = ""
                    elif hasattr(value, "isoformat"):
                        clean_run[key] = value.isoformat()
                    else:
                        clean_run[key] = value
            
            runs.append(clean_run)
            
        return {"runs": runs}

    except Exception as e:
        print(f"Error searching runs in MLflow for '{experiment_name}': {str(e)}")
        # If it's a known error from mlflow or pandas, we catch it
        raise HTTPException(status_code=500, detail=f"Error searching runs in MLflow: {str(e)}")