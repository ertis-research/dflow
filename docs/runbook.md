# DFlow API - Runbook & Troubleshooting Guide

This runbook provides operational procedures, health checks, and troubleshooting steps for the DFlow API and its underlying infrastructure (Kubernetes, MinIO, and MLflow).

---

## 1. Architecture Overview

When a user submits an experiment, the following sequence occurs:
1. **API (FastAPI)** receives the CSV and JSON config.
2. **MinIO** stores the CSV dataset.
3. **Kubernetes** spins up an ephemeral `Job` using the `ludwigai/ludwig` image.
4. **Ludwig Pod** downloads data, trains the model, and communicates with MLflow.
5. **MLflow** receives metrics and registers the trained model.

---

## 2. Health Checks

### 2.1 Check API Status
```bash
curl -X GET http://localhost:8000/
# Expected: {"message": "Welcome to the dflow API..."}
```

### 2.2 Check Kubernetes Namespace
Ensure the target namespace exists and jobs are running:
```bash
kubectl get namespace api-dlflow
kubectl get pods -n api-dlflow
```

### 2.3 Check MinIO & MLflow Connectivity
Ensure the environment variables in `.env` are pointing to reachable IPs/Domains.
* **MinIO:** Access the MinIO Web Console using the URL in `MINIO_URL`.
* **MLflow:** Access the MLflow Web UI using the URL in `MLFLOW_TRACKING_URI`.

---

## 3. Troubleshooting & Common Issues

### Issue 1: Kubernetes Pod Fails Immediately (Status: `Error`)

**Symptoms:** The training pod enters an `Error` state shortly after creation.

**Diagnosis:**
1. Find the failed pod:
   ```bash
   kubectl get pods -n api-dlflow --sort-by=.metadata.creationTimestamp
   ```
2. Read the pod logs:
   ```bash
   kubectl logs <pod-name> -n api-dlflow
   ```

**Common Causes & Fixes:**
* **`KeyError: 'column_name'`**: The column names in the `config_json` do not exactly match the columns in the uploaded CSV. Check for typos, case sensitivity, or trailing spaces.
* **`MinioException: S3 API Request...`**: MinIO is unreachable or credentials in `.env` are incorrect.
* **`mlflow.exceptions.RestException`**: The MLflow tracking URI is incorrect or the server is down.

### Issue 2: Training Succeeds but Model is Not in MLflow

**Symptoms:** Pod logs show `SUCCESS: Model trained`, but the model is missing from the MLflow Model Registry.

**Diagnosis:**
1. Check the pod logs for MLflow warnings during the registration phase.
2. Verify that `model_name` and `experiment_name` were correctly provided in the Swagger UI.

**Fix:**
Ensure the `MLFLOW_TRACKING_URI` in your `.env` file is accessible from *inside* the Kubernetes cluster. If using a local IP (like `192.168.x.x`), verify the cluster nodes can route to it.

### Issue 3: Pod Stuck in `Pending` or `ImagePullBackOff`

**Symptoms:** The pod is created but never starts running.

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n api-dlflow
```

**Common Causes & Fixes:**
* **Insufficient Resources:** The Kubernetes node doesn't have enough CPU/RAM for the Ludwig image.
* **Network Issues:** The node cannot pull the `ludwigai/ludwig:latest` image from Docker Hub.

---

## 4. Maintenance Operations

### 4.1 Cleaning up Old Kubernetes Jobs
To prevent the cluster from filling up with completed or failed jobs, periodically clean them up:
```bash
# Delete all completed jobs
kubectl delete jobs -n api-dlflow --field-selector status.successful=1

# Delete all failed jobs
kubectl delete jobs -n api-dlflow --field-selector status.failed=1
```

### 4.2 Rebuilding the Local Environment
If you need to reset your local Python environment:
```bash
rm -rf venv/
python -m venv venv
# Activate it, then:
pip install -r requirements.txt
```
