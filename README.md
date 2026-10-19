# DFlow API - MLOps Orchestrator

DFlow API is an MLOps orchestration engine built to automate the Machine Learning model lifecycle. It provides a unified interface to upload datasets, define declarative training configurations, and delegate heavy computational workloads to a Kubernetes cluster. The system natively integrates with MLflow to ensure complete traceability, metrics tracking, and version control.

---

## Key Features

* **S3-Compatible Storage:** Secure dataset ingestion and management via MinIO.
* **Kubernetes Orchestration:** Automated provisioning of isolated, ephemeral Jobs for individual training experiments using the official Ludwig image.
* **Declarative Auto-ML:** Integrates with Ludwig AI to enable model training through JSON configuration schemas, abstracting boilerplate deep learning code.
* **Unified Experiment Management:** New endpoints to list, track, and delete experiments from both local state and MLflow history.
* **Detailed Run Tracking:** Retrieve metrics, parameters, and tags for individual training runs directly from MLflow.
* **Asynchronous Execution:** Non-blocking API design that offloads resource-intensive tasks to the cluster backend.
* **CORS Support:** Enabled for seamless integration with frontend applications.

---

## Tech Stack

* **API Framework:** FastAPI (Python)
* **Data Storage:** MinIO
* **Container Orchestration:** Kubernetes (K8s)
* **Machine Learning Engine:** Ludwig AI
* **Tracking & Registry:** MLflow

---

## Prerequisites

To run this project, the following dependencies must be configured in your environment:

1. Python 3.10+
2. Access to a Kubernetes cluster (with a valid `~/.kube/config` file).
3. An operational MinIO server with access credentials.
4. An operational MLflow Tracking server.

---

## Local Installation and Deployment

### 1. Clone the repository

```bash
git clone https://github.com/ertis-research/dflow.git
cd dflow
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

#### On Windows

```powershell
.\venv\Scripts\activate
```

#### On Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
MINIO_URL=minio.your-domain.com
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
BUCKET_NAME=your_bucket_name
KUBERNETES_CONFIG_PATH=~/.kube/config
KUBERNETES_NAMESPACE=api-dlflow
MLFLOW_TRACKING_URI=http://your-mlflow-ip:63163
```

### 5. Start the API

```bash
python -m uvicorn src.main:app --reload --port 8000
```

---

## API Endpoints (Prefix: `/api/mlops`)

The API follows a modular path structure. Interactive documentation is available at `http://localhost:8000/docs`.

### 1. List Experiments

* **Method:** `GET`
* **Path:** `/api/mlops/experiments`
* **Description:** Returns a merged list of experiments from the local tracking database (active jobs) and MLflow history.

### 2. Create Experiment

* **Method:** `POST`
* **Path:** `/api/mlops/experiments`
* **Description:** Initializes a new training job.
* **Multipart Fields:**
    | Field             | Description                                                        |
    | ----------------- | ------------------------------------------------------------------ |
    | `csv_file`        | Dataset file to be processed (e.g. `data.csv`)                     |
    | `model_type`      | Internal reference algorithm (e.g. `Ludwig_Test`)                  |
    | `config_json`     | Stringified JSON object containing Ludwig configuration parameters |
    | `dependencies`    | (Optional) JSON list or comma-separated string of dependencies     |
    | `experiment_name` | Logical grouping name in MLflow (Default: `dflow-default-experiment`) |
    | `model_name`      | Final model artifact name in the MLflow Model Registry (Default: `ludwig_model`) |

#### Example cURL Request

```bash
curl -X POST "http://localhost:8000/api/mlops/experiments" \
  -F "csv_file=@data.csv" \
  -F "model_type=Ludwig_Test" \
  -F "experiment_name=Predictive_Maintenance" \
  -F "model_name=Engine_Predictor" \
  -F 'config_json={
    "input_features": [{"name": "rpm", "type": "numerical"}],
    "output_features": [{"name": "imminent_failure", "type": "category"}],
    "trainer": {"epochs": 10}
  }'
```

### 3. Delete Experiment

* **Method:** `DELETE`
* **Path:** `/api/mlops/experiments/{experiment_name}`
* **Description:** Removes the experiment from both the local database and the MLflow Tracking Server.

### 4. Get Runs

* **Method:** `GET`
* **Path:** `/api/mlops/experiments/{experiment_name}/runs`
* **Description:** Retrieves all training runs associated with the experiment name from MLflow, including metrics and parameters.

---

## Architecture Workflow

```text
1. API receives dataset + training configuration.
2. CSV file is uploaded to MinIO.
3. Kubernetes provisions a Ludwig training Job.
4. Pod downloads dataset and executes training.
5. MLflow logs metrics and registers the trained model.
```

### Detailed Execution Flow

1. **Ingestion:** API receives the dataset and training JSON. The CSV is uploaded to MinIO under `<experiment_id>/filename.csv`.
2. **Orchestration:** A Kubernetes Job is created using the `ludwigai/ludwig:latest` image.
3. **Training:** The Pod executes a generated Python script that:
    - Downloads the dataset from MinIO.
    - Initializes the `LudwigModel`.
    - Starts the training pipeline.
4. **Traceability:** Upon completion, training statistics and the model artifact are automatically logged to MLflow via integrated callbacks.

---

## Project Structure

```text
.
├── src/
│   ├── main.py        # API Entry point and route definitions
│   ├── k8s_utils.py   # Kubernetes Job orchestration logic
│   ├── minio_utils.py # Storage interaction utilities
│   └── database.py    # Local state tracking (In-memory)
├── data/              # Sample datasets
├── docs/              # Documentation and runbooks
├── requirements.txt
├── .env
└── README.md
```

---

## Future Improvements

* Add authentication and role-based access.
* Support distributed training workloads.
* Add GPU scheduling support in Kubernetes.
* Integrate hyperparameter optimization.
* Add experiment monitoring dashboard.

---

## License

This project is licensed under the MIT License.
