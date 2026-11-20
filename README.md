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
py -3.12 -m venv .venv
```

#### On Windows

```powershell
.\.venv\Scripts\activate
```

#### On Linux / macOS

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` in the project root and fill in your values:

```bash
cp .env.example .env
```

The minimum configuration looks like this:

```env
MINIO_URL=minio.your-domain.com
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
BUCKET_NAME=your_bucket_name
KUBERNETES_CONFIG_PATH=~/.kube/config
KUBERNETES_NAMESPACE=<YOUR_K8S_NAMESPACE>
MLFLOW_TRACKING_URI=http://your-mlflow-ip:63163

# Training store (see "Training store" below)
DB_BACKEND=sqlite
SQLITE_PATH=./data/dflow.db
```

All configuration comes from environment variables. If a required variable is missing, dflow does not start and the log names the variable.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `MINIO_URL`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | yes | | MinIO endpoint and credentials |
| `BUCKET_NAME` | yes | | Bucket for the uploaded datasets |
| `MINIO_SECURE` | no | `true` | Use HTTPS for MinIO |
| `KUBERNETES_NAMESPACE` | yes | | Namespace where training Jobs run |
| `KUBERNETES_CONFIG_PATH` | no | `~/.kube/config` | Kubeconfig, used only outside the cluster |
| `MLFLOW_TRACKING_URI` | no | | MLflow server. Without it, only local trainings are tracked |
| `TRAINING_IMAGE` | no | `ludwigai/ludwig:latest` | Image of the training Jobs |
| `TRAINING_JOB_TTL_SECONDS` | no | `60` | Kubernetes removes finished Jobs after this time |
| `TRAINING_JOB_BACKOFF_LIMIT` | no | `1` | Retries of a failed training pod |
| `STATUS_POLL_INTERVAL_SECONDS` | no | `10` | How often training statuses are refreshed |
| `LAUNCH_TIMEOUT_SECONDS` | no | `300` | A training still `queued`/`launching_job` after this time is marked `failed` |
| `DB_BACKEND` | no | `sqlite` | `sqlite` or `postgres` |
| `SQLITE_PATH` | no | `/data/dflow.db` | SQLite file (when `DB_BACKEND=sqlite`) |
| `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | if `postgres` | | PostgreSQL connection |
| `POSTGRES_PORT` | no | `5432` | PostgreSQL port |

### Training store

dflow keeps its trainings in a SQL database, so they survive restarts. Choose the backend with `DB_BACKEND`:

* **`sqlite`** (default): a file inside the dflow container, with no extra services. In Kubernetes, `k8s/deployment.yaml` mounts the `dflow-data` volume at `/data`. Without a volume (for example `docker run` without `-v`), data is lost when the container is removed. Only one replica can be used.
* **`postgres`**: an external PostgreSQL server (the database must exist; dflow creates its table at startup). Use it to share the store or to run several replicas. In Kubernetes, create the optional `dflow-postgres` Secret, which the deployment loads automatically:

```bash
kubectl -n <YOUR_K8S_NAMESPACE> create secret generic dflow-postgres \
  --from-literal=DB_BACKEND=postgres \
  --from-literal=POSTGRES_HOST=<host> --from-literal=POSTGRES_DB=<db> \
  --from-literal=POSTGRES_USER=<user> --from-literal=POSTGRES_PASSWORD=<password>
```

### 5. Start the API

```bash
python -m uvicorn src.main:app --reload --port 8000
```

### Running tests

The tests use dummy settings and mocks, so they need neither a `.env` file nor Postgres, MinIO, MLflow or Kubernetes.

```bash
python -m venv .venv            # or reuse the one from step 2
# activate it (Windows: .venv\Scripts\activate, Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
pytest
```

## Docker and Kubernetes Deployment

### 1. Build docker image

```bash
docker build -t <YOUR_REGISTRY>/<YOUR_PROJECT>/dflow:latest ./
```

### 2. Push docker image into registry

```bash
docker push <YOUR_REGISTRY>/<YOUR_PROJECT>/dflow:latest
```

After the image is pushed, it can be initialized using docker or it can be deployed on Kubernetes if wanted.

### 3. Deployment

Depending on the user's needs, it can be either locally deployed using Docker or use a Kubernetes pod:

#### 3.1. Docker deployment

```bash
docker run -p 8000:8000 <YOUR_REGISTRY>/<YOUR_PROJECT>/dflow:latest
```

#### 3.2. Kubernetes deployment

`k8s/deploy.sh` deploys dflow from your `.env` file (see `.env.example`). It:
1. creates or updates the `dflow-minio-credentials` Secret with dflow's configuration;
2. creates or updates the `dflow-registry-creds` pull Secret, if `REGISTRY_*` are set;
3. applies `k8s/rbac.yaml` and `k8s/deployment.yaml` in the `KUBERNETES_NAMESPACE` namespace, using `DFLOW_IMAGE` as the image;
4. restarts dflow so it reads the new configuration and pulls the image again.

```bash
DRY_RUN=1 k8s/deploy.sh    # validate against the cluster, change nothing
k8s/deploy.sh              # deploy (or k8s/deploy.sh path/to/other.env)
```

It needs bash and `kubectl` (on Windows, run it from Git Bash). Credentials never go into git or into the manifests. The Secrets are written with `create`/`replace`, not `kubectl apply`, so their values are not copied into the `last-applied-configuration` annotation.

What it takes from `.env`:

| Variable | Used for |
| --- | --- |
| `KUBERNETES_NAMESPACE` | The namespace where dflow is deployed and where it launches the training Jobs |
| `DFLOW_IMAGE` | The dflow image to deploy (the one pushed in the previous steps) |
| `REGISTRY_SERVER`, `REGISTRY_USERNAME`, `REGISTRY_PASSWORD` | The `dflow-registry-creds` pull Secret. Required the first time: `deploy.sh` fails if `dflow-registry-creds` does not exist in the namespace and these are not set. An older `model-registry-creds` Secret (e.g. the one in namespaces provisioned by Bentloy) is no longer reused. Once `dflow-registry-creds` exists, you can omit them to keep it |
| Everything else | The `dflow-minio-credentials` Secret, i.e. dflow's configuration. `SQLITE_PATH` and `KUBERNETES_CONFIG_PATH` are local-only and are not copied: in the cluster dflow uses `/data/dflow.db` (the persistent volume) and its service account |

**Registry credentials.** They need **pull permission on the registry project where the image is**; otherwise the pod stays in `ImagePullBackOff` with `401 Unauthorized`. Harbor robot accounts are usually limited to their own project. For example, `robot$bentloy-test-2+bentloy` can pull from `docker.ertis.uma.es/bentloy-test-2/...` but not from `docker.ertis.uma.es/opentwinsv2/...`. Push the image to the robot's project, or give the robot pull permission on the image's project. Use a robot account, not a personal user.

**Without the script.** The manifests need both Secrets to exist and `image:` to be edited by hand:

```bash
kubectl -n <ns> create secret docker-registry dflow-registry-creds   --docker-server=<registry> --docker-username='<robot>' --docker-password='<token>'
kubectl -n <ns> create secret generic dflow-minio-credentials --from-env-file=<env file without SQLITE_PATH>
kubectl apply -n <ns> -f k8s/rbac.yaml -f k8s/deployment.yaml
```

Pod status can be checked using the command:

```bash
kubectl get pods -n <YOUR_K8S_NAMESPACE>
```

Once deployed, Kubernetes will expose the API through a Service. To find the API adress and port to access the API, run:

```bash
kubectl get svc -n <YOUR_K8S_NAMESPACE>
```

---

## API Endpoints (Prefix: `/api/mlops`)

The API follows a modular path structure. Interactive documentation is available at `http://localhost:8000/docs`.

### 1. List Experiments

* **Method:** `GET`
* **Path:** `/api/mlops/experiments`
* **Description:** Returns one entry per experiment (model), merging local trainings and MLflow history. `GET /api/mlops/experiments/{experiment_name}` returns a single experiment with the same fields.

#### Training status

Each experiment has a `status` from this closed list and, when it failed, a `status_reason`:

| `status` | Meaning |
| --- | --- |
| `queued` | Request accepted and dataset stored; the Kubernetes Job has not been requested yet |
| `launching_job` | The Kubernetes Job is being created |
| `training` | The Job is running (or its pod is pending). `status_reason` explains a stuck pod, e.g. `ImagePullBackOff` |
| `completed` | The training finished and the model is registered in MLflow |
| `failed` | The training failed. `status_reason` gives the cause: the last `ERROR` line of the pod logs, the Job failure message, or e.g. `The training job no longer exists` |
| `cancelled` | Returned by the cancel endpoint. It only appears in listings while the cleanup of a cancellation failed and is waiting to be retried (`status_reason` says why) |

dflow refreshes the status of active trainings in the background from the Kubernetes Job and the MLflow run. When the latest training of an experiment has a finished MLflow run, the experiment is `completed`, so the listing never contradicts MLflow.

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

Returns `409 Conflict` if the experiment already has a training in progress.

### 3. Delete Experiment

* **Method:** `DELETE`
* **Path:** `/api/mlops/experiments/{experiment_name}`
* **Description:** Removes the experiment from the local database and the MLflow Tracking Server, and deletes the uploaded datasets (training CSVs) of all its trainings from MinIO. If a training is in progress, its Kubernetes Job is stopped first.
* **Errors:** `404` if the experiment does not exist. `502` if the Kubernetes Job of the training in progress could not be deleted, or if the datasets could not be deleted from MinIO; nothing else is deleted in that case, so the request can be retried.

### 4. Cancel Training

* **Method:** `POST`
* **Path:** `/api/mlops/experiments/{experiment_name}/cancel`
* **Description:** Stops the training in progress (`queued`, `launching_job` or `training`) and removes it:
    - deletes the Kubernetes Job and its pods;
    - marks its MLflow run as `KILLED` and deletes it, with any model version registered from it;
    - deletes its dataset from MinIO.

  If the model has no previous versions, the whole experiment is deleted (as with `DELETE`). Otherwise only this training is removed and the existing versions are kept.
* **Responses:** `404` if there is no training in progress for that name. `200` otherwise:

```json
{
  "message": "Training 'ab12cd34' of 'X' cancelled; experiment deleted because it had no versions",
  "experiment_name": "X",
  "training_id": "ab12cd34",
  "status": "cancelled",
  "experiment_deleted": true,
  "cleanup_complete": true,
  "deleted": {"k8s_job": "dflow-train-job-ab12cd34", "mlflow_runs": [], "model_versions": [], "minio_objects": 1, "local_record": true},
  "errors": []
}
```

If a step fails (for example, MinIO unreachable), `cleanup_complete` is `false` and `errors` says which one. In that case:
- the training stays as `cancelled` and the experiment is not deleted;
- dflow retries the cleanup automatically, and calling the endpoint again also retries it.

Cancelling only ever deletes the cancelled training itself. If a new training of the same model has already started, it is not touched.

### 5. Get Runs

* **Method:** `GET`
* **Path:** `/api/mlops/experiments/{experiment_name}/runs`
* **Description:** Retrieves all training runs associated with the experiment name from MLflow, including metrics and parameters.

### 6. Get Training Logs

* **Method:** `GET`
* **Path:** `/api/mlops/experiments/{experiment_name}/logs`
* **Query param:** `tail_lines` (default `500`, range `1`-`5000`): number of log lines to return, counted from the end.
* **Description:** Returns the logs of the training pod of the experiment (the training in progress, or else the latest one). It is intended to show the training progress live in the OpenTwins UI / Grafana, by polling it.
* **Responses:**
    - `200`: the logs, or `available: false` with a `reason` (see below).
    - `404`: the experiment does not exist.
    - `502`: Kubernetes returned an error while reading the logs.

```json
{
  "experiment_name": "X",
  "training_id": "ab12cd34",
  "job_name": "dflow-train-job-ab12cd34",
  "pod_name": "dflow-train-job-ab12cd34-xk2lp",
  "status": "training",
  "pod_phase": "Running",
  "logs": "Epoch 1/10 ...\n",
  "available": true,
  "reason": null
}
```

Cases that return `available: false` (with `logs` empty):
- `"The training has not started yet."`: the training is `queued` or `launching_job`.
- `"The training pod is starting."`: the pod exists but its container has not started yet (the pod waiting reason is appended in parentheses when known).
- `"The training pod no longer exists; its logs are not kept."`: the Job or its pod has been deleted.
- `"This model was not trained by dflow, or its training is no longer kept."`: the experiment only exists in MLflow.

> **Note:** because of the TTL (`TRAINING_JOB_TTL_SECONDS`), the Job and its pods are deleted after the training finishes, so the logs of finished trainings are not kept.

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
2. **Orchestration:** A Kubernetes Job is created using the `TRAINING_IMAGE` image (`ludwigai/ludwig:latest` by default).
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
│   ├── main.py            # API Entry point and route definitions
│   ├── config.py          # Settings read from environment variables
│   ├── k8s_utils.py       # Kubernetes Job orchestration logic
│   ├── minio_utils.py     # Storage interaction utilities
│   ├── mlflow_utils.py    # MLflow helpers (runs of a training, cleanup)
│   ├── training_status.py # Background refresh of training statuses
│   ├── schemas.py         # Schemas and the training status vocabulary
│   ├── service.py         # Service for core API functionalities
│   └── database.py        # Training store (SQLite or PostgreSQL)
├── data/              # Sample datasets
|   └── data.csv       # Sample dataset for README example POST
├── docs/              # Documentation and runbooks
├── k8s/               # Kubernetes manifests and deploy script
│   ├── deploy.sh       # Deploys dflow from .env (Secrets + manifests)
│   ├── deployment.yaml # Deployment, Service and SQLite volume
│   └── rbac.yaml       # Permissions for training Jobs and pod logs
├── requirements.txt
├── .env.example       # Configuration template (copy to .env)
├── .dockerignore
├── Dockerfile
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
