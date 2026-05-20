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
    """Returns the list of all running and completed experiments."""
    return {"experiments": fake_database}

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

@app.delete("/api/mlops/experiment/{experiment_id}")
def delete_experiment(experiment_id: str):
    """Deletes an experiment by its unique ID from the local DB and MLflow."""
    if experiment_id not in fake_database:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    # 1. Recuperar el nombre del experimento ANTES de borrarlo de la base local
    exp_data = fake_database[experiment_id]
    mlflow_exp_name = exp_data.get("experiment_name")
    
    # 2. Intentar borrar en MLflow (operación de red muy rápida)
    if mlflow_exp_name and mlflow_uri:
        try:
            # Buscar el experimento en MLflow por su nombre
            mlflow_exp = mlflow.get_experiment_by_name(mlflow_exp_name)
            
            # Si existe y no está ya borrado, lo eliminamos
            if mlflow_exp and mlflow_exp.lifecycle_stage != "deleted":
                mlflow.delete_experiment(mlflow_exp.experiment_id)
                print(f"MLflow: Experimento '{mlflow_exp_name}' eliminado.")
        except Exception as e:
            # Si MLflow falla (ej. servidor caído), lo capturamos para que no 
            # rompa la API y permita borrar al menos el registro local
            print(f"Error borrando en MLflow: {str(e)}")

    # 3. Borrar de la base de datos local (fake_database)
    del fake_database[experiment_id]
    
    return {"message": f"Experiment {experiment_id} successfully deleted"}