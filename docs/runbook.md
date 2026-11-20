# DFlow API - Runbook & Troubleshooting Guide

This runbook provides operational procedures, health checks, and troubleshooting steps for the DFlow API and its underlying infrastructure (Kubernetes, MinIO, and MLflow).

---

## 1. Architecture Overview

When a user submits an experiment, the following sequence occurs:
1. **API (FastAPI)** receives the CSV and JSON config.
2. **MinIO** stores the CSV dataset.
3. **Kubernetes** spins up an ephemeral `Job` using the `TRAINING_IMAGE` image (`ludwigai/ludwig:latest` by default).
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
kubectl get namespace <YOUR_K8S_NAMESPACE>
kubectl get pods -n <YOUR_K8S_NAMESPACE>
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
   kubectl get pods -n <YOUR_K8S_NAMESPACE> --sort-by=.metadata.creationTimestamp
   ```
2. Read the pod logs:
   ```bash
   kubectl logs <pod-name> -n <YOUR_K8S_NAMESPACE>
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
kubectl describe pod <pod-name> -n <YOUR_K8S_NAMESPACE>
```

**Common Causes & Fixes:**
* **Insufficient Resources:** The Kubernetes node doesn't have enough CPU/RAM for the Ludwig image.
* **Network Issues:** The node cannot pull the training image (`TRAINING_IMAGE`).

### Issue 4: Experiment shows `failed` with a reason

The `status_reason` field says why. Common values:
* **`ERROR IN SETUP PHASE: …` / `ERROR DURING TRAINING: …` / `ERROR DURING MLFLOW LOGGING: …`**: last error line printed by the training pod (see Issue 1 and Issue 2).
* **`The training job no longer exists`**: the Job was deleted (by hand, or by its TTL while dflow was down) and no MLflow run was registered.
* **`Training finished but the model was not registered in MLflow`**: the Job succeeded but no run reached MLflow. Check `MLFLOW_TRACKING_URI` from inside the cluster.
* **`The training job was never launched`**: the training stayed `queued`/`launching_job` longer than `LAUNCH_TIMEOUT_SECONDS`.

If reasons are always generic (e.g. `BackoffLimitExceeded`), dflow cannot read pod logs. Check the RBAC permissions:
```bash
kubectl auth can-i get pods/log -n <YOUR_K8S_NAMESPACE> --as=system:serviceaccount:<YOUR_K8S_NAMESPACE>:default
```

### Issue 5: dflow pod in `ImagePullBackOff` with `401 Unauthorized`

The `dflow-registry-creds` pull secret is missing, or its credentials cannot read the registry project of the image.

**Diagnosis:**
```bash
# Which image and which pull secret the deployment uses
kubectl -n <YOUR_K8S_NAMESPACE> get deploy dflow \
  -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}{.spec.template.spec.imagePullSecrets}{"\n"}'
# Does the secret exist?
kubectl -n <YOUR_K8S_NAMESPACE> get secret dflow-registry-creds
```

**Fix:** the image must be in a registry project that the secret's account can pull from. Harbor robot accounts are usually limited to their own project. Push the image to that project, or give the robot pull permission on the image's project. Logging in with your own user on your machine (`docker login`) does not help: the cluster pulls with the secret. `k8s/deploy.sh` creates the secret from `REGISTRY_*` in `.env`; see README, "Kubernetes deployment".

### Issue 6: dflow pod in `CreateContainerConfigError`

A Secret referenced by the deployment does not exist, normally `dflow-minio-credentials`. Check it with `kubectl -n <YOUR_K8S_NAMESPACE> describe pod <dflow-pod>`, then run `k8s/deploy.sh`, which creates it from `.env` (README, "Kubernetes deployment").

### Issue 7: dflow does not start

The log names the problem: a missing environment variable (`Missing required environment variable: …`), an unreachable PostgreSQL, or MinIO. With `DB_BACKEND=sqlite`, check that the `dflow-data` volume is bound (`kubectl get pvc -n <YOUR_K8S_NAMESPACE>`).

---

## 4. Maintenance Operations

### 4.1 Cleaning up Old Kubernetes Jobs
Finished training Jobs are removed automatically after `TRAINING_JOB_TTL_SECONDS` (60 by default). To stop a running training, use `POST /api/mlops/experiments/{name}/cancel`. To clean up leftover jobs by hand:
```bash
# Delete all completed jobs
kubectl delete jobs -n <YOUR_K8S_NAMESPACE> --field-selector status.successful=1

# Delete all failed jobs
kubectl delete jobs -n <YOUR_K8S_NAMESPACE> --field-selector status.failed=1
```

### 4.2 Rebuilding the Local Environment
If you need to reset your local Python environment:
```bash
rm -rf venv/
python -m venv venv
# Activate it, then:
pip install -r requirements.txt
```

### 4.3 Migrating to dflow- prefixed names
Every Kubernetes object dflow creates now has a name starting with `dflow-`: Secrets `dflow-registry-creds` and `dflow-minio-credentials`, Role `dflow-job-creator-role`, RoleBinding `dflow-job-creator-binding`, and training Jobs `dflow-train-job-<id>`. The Deployment is now called `dflow` (it was `dflow-deployment`).

**Before** deploying this version, delete the old Deployment: both select the pods labelled `app=dflow` (the Service would route to both) and the `dflow-data` volume can only be mounted by one pod at a time:
```bash
kubectl -n <namespace> delete deployment dflow-deployment
```

After deploying this version with `k8s/deploy.sh` (which needs `REGISTRY_*` in `.env` to create `dflow-registry-creds`), the old objects are left orphaned. Remove them:
```bash
kubectl -n <namespace> delete secret minio-credentials
kubectl -n <namespace> delete role job-creator-role
kubectl -n <namespace> delete rolebinding job-creator-binding
kubectl -n <namespace> delete secret model-registry-creds
```

**Warning:** do NOT delete `model-registry-creds` if other services in the namespace still use it (e.g. Bentoml).

Jobs already created as `train-job-*` keep working (dflow uses the `job_name` stored for each training) and are removed by their TTL.
