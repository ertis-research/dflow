# DFlow API - MLOps Orchestrator

DFlow API is an MLOps orchestration engine built to automate the Machine Learning model lifecycle. It provides a unified interface to upload datasets, define declarative training configurations, and delegate heavy computational workloads to a Kubernetes cluster. The system natively integrates with MLflow to ensure complete traceability, metrics tracking, and version control.

---

## Key Features

* **S3-Compatible Storage:** Secure dataset ingestion and management via MinIO.
* **Kubernetes Orchestration:** Automated provisioning of isolated, ephemeral Jobs for individual training experiments.
* **Declarative Auto-ML:** Integrates with Ludwig AI to enable model training through JSON configuration schemas, abstracting boilerplate deep learning code.
* **Experiment Tracking & Versioning:** Native integration with MLflow for automatic logging of hyperparameters, training metrics, and model artifact registration.
* **Asynchronous Execution:** Non-blocking API design that offloads resource-intensive tasks to the cluster backend.

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
git clone https://github.com/your-username/dflow.git
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

## API Usage (Swagger UI)

Once the server is running, the interactive API documentation is available at:

```text
http://localhost:8000/docs
```

---

## Main Endpoint: `POST /experiment`

This endpoint receives the configuration payload and initializes the Kubernetes Job.

It requires the following fields in `multipart/form-data` format:

| Field             | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| `csv_file`        | Dataset file to be processed (e.g. `data.csv`)                     |
| `model_type`      | Internal reference algorithm (e.g. `Ludwig_Test`)                  |
| `experiment_name` | Logical grouping name in MLflow (e.g. `Predictive_Maintenance`)    |
| `model_name`      | Final model artifact name in the MLflow Model Registry             |
| `config_json`     | Stringified JSON object containing Ludwig configuration parameters |

### Example `config_json`

```json
{
  "input_features": [
    {
      "name": "rpm",
      "type": "numerical"
    },
    {
      "name": "vibration",
      "type": "numerical"
    }
  ],
  "output_features": [
    {
      "name": "imminent_failure",
      "type": "category"
    }
  ],
  "trainer": {
    "epochs": 10
  }
}
```

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/experiment" \
  -F "csv_file=@data.csv" \
  -F "model_type=Ludwig_Test" \
  -F "experiment_name=Predictive_Maintenance" \
  -F "model_name=Engine_Predictor" \
  -F 'config_json={
    "input_features": [
      {"name": "rpm", "type": "numerical"},
      {"name": "vibration", "type": "numerical"}
    ],
    "output_features": [
      {"name": "imminent_failure", "type": "category"}
    ],
    "trainer": {
      "epochs": 10
    }
  }'
```

---

## Architecture Workflow

```text
1. API receives dataset + training configuration.
2. CSV file is uploaded to MinIO.
3. Kubernetes provisions a Ludwig training Job.
4. Pod downloads dataset and executes training.
5. MLflow logs metrics and registers the trained model.
```

### Detailed Flow

1. The API receives the dataset and the JSON training configuration.
2. The CSV file is securely uploaded to the MinIO bucket at the designated path:

```text
<experiment_id>/filename.csv
```

3. The Kubernetes API provisions a Job using the official Ludwig image:

```text
ludwigai/ludwig
```

4. The spawned Pod:

* Downloads the dataset.
* Parses the configuration.
* Executes the training pipeline using the Ludwig Python API.

5. A built-in callback:

* Sends real-time metrics to MLflow.
* Registers the trained model artifact in the MLflow Model Registry.

---

## Project Structure

```text
.
├── src/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   ├── kubernetes/
│   └── storage/
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
