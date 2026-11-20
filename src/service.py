import asyncio
import datetime
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, cast
import uuid
import pandas as pd
from fastapi import BackgroundTasks, UploadFile
import mlflow
from kubernetes.client.rest import ApiException
from mlflow.entities import Experiment as MlflowExperiment

from src.cancellation import cleanup_cancelled_training
from src.config import get_settings
from src.database import ActiveTrainingExists, get_store
from src.k8s_utils import (
    CONTAINER_NAME,
    _job_pods,
    core_api,
    delete_training_job,
    get_pod_waiting_reason,
    job_name_for,
    train_model_job,
)
from src.minio_utils import delete_training_data, upload_csv_to_minio
from src.mlflow_utils import (
    TRAINING_ID_TAG,
    get_active_experiment,
    prepare_experiment_for_training,
)
from src.schemas import ACTIVE_STATUSES, Experiment, PaginatedResult, PaginatedMetaData, Run, TrainingLogs
from src.training_status import reconcile_training

REASON_NO_SUCCESSFUL_RUN = "No successful run in MLflow"

__all__ = ["DFlowService", "ActiveTrainingExists", "NoActiveTraining"]


class NoActiveTraining(Exception):
    """Raised when cancelling an experiment that has no training in progress."""


@dataclass
class _RunSummary:
    """What the experiment listing needs to know about the runs of one MLflow experiment."""
    latest_status: Optional[str] = None
    latest_start: int = -1
    has_finished: bool = False
    finished_training_ids: set = field(default_factory=set)

    def add(self, status: str, start_time: Optional[int], training_id: Optional[str]) -> None:
        start = int(start_time or 0)
        if start >= self.latest_start:
            self.latest_start = start
            self.latest_status = status
        if status == "FINISHED":
            self.has_finished = True
            if training_id:
                self.finished_training_ids.add(training_id)


class DFlowService:
    def __init__(self) -> None:
        pass

    # ---- status resolution (local training vs MLflow) ------------------------

    @staticmethod
    def _resolve_status(training: Optional[Dict[str, Any]], summary: Optional[_RunSummary]) -> tuple:
        """Single source of truth for the experiment status. Returns (status, status_reason).

        - The latest local training decides while it is active, failed or completed...
        - ...except that a FINISHED MLflow run of that training means it is completed.
        - Without local trainings, MLflow decides: RUNNING -> training, any FINISHED -> completed.
        """
        if training is not None:
            if training["status"] == "cancelled":
                # Only visible while the cleanup of a cancellation is pending a retry.
                return "cancelled", training.get("status_reason")
            if summary is not None and training["training_id"] in summary.finished_training_ids:
                return "completed", None
            return training["status"], training.get("status_reason")

        if summary is None:
            return "failed", REASON_NO_SUCCESSFUL_RUN
        if summary.latest_status in ("RUNNING", "SCHEDULED"):
            return "training", None
        if summary.has_finished:
            return "completed", None
        return "failed", REASON_NO_SUCCESSFUL_RUN

    @staticmethod
    def _build_experiment(
        name: str,
        training: Optional[Dict[str, Any]],
        mlflow_exp: Optional[MlflowExperiment],
        summary: Optional[_RunSummary],
    ) -> Experiment:
        status, reason = DFlowService._resolve_status(training, summary)
        t = training or {}
        return Experiment(
            experiment_name=name,
            source="mlflow" if mlflow_exp is not None else "local",
            status=status,
            status_reason=reason,
            internal_id=t.get("training_id"),
            mlflow_id=mlflow_exp.experiment_id if mlflow_exp is not None else None,
            artifact_location=mlflow_exp.artifact_location if mlflow_exp is not None else None,
            model=t.get("model_type"),
            model_name=t.get("model_name"),
            config=t.get("config"),
            dependencies=t.get("dependencies") or [],
            csv_filename=t.get("csv_filename"),
            created_at=t.get("created_at"),
        )

    @staticmethod
    def _summarize_runs(experiment_ids: List[str]) -> Dict[str, _RunSummary]:
        summaries: Dict[str, _RunSummary] = {eid: _RunSummary() for eid in experiment_ids}
        if not experiment_ids:
            return summaries
        runs = mlflow.search_runs(experiment_ids=experiment_ids, output_format="list")
        for run in cast(list, runs):
            summary = summaries.setdefault(run.info.experiment_id, _RunSummary())
            summary.add(run.info.status, run.info.start_time, run.data.tags.get(TRAINING_ID_TAG))
        return summaries

    # ---- queries -------------------------------------------------------------

    def get_all_experiments(self, mlflow_on: bool, page: int = 0, pageSize: int = 20, search: str = '') -> PaginatedResult[Experiment]:
        # Latest training per experiment name (list_all is newest first).
        latest: Dict[str, Dict[str, Any]] = {}
        for training in get_store().list_all():
            latest.setdefault(training["experiment_name"], training)

        mlflow_exps: Dict[str, MlflowExperiment] = {}
        summaries: Dict[str, _RunSummary] = {}
        if mlflow_on:
            try:
                for exp in mlflow.search_experiments():
                    if exp.name != "Default":
                        mlflow_exps[exp.name] = exp
                summaries = self._summarize_runs([e.experiment_id for e in mlflow_exps.values()])
            except Exception as e:
                print(f"Error reading from MLflow: {str(e)}")

        all_experiments: List[Experiment] = []
        # Experiments with local trainings first (newest first), then MLflow-only ones.
        for name, training in latest.items():
            exp = mlflow_exps.get(name)
            summary = summaries.get(exp.experiment_id) if exp is not None else None
            all_experiments.append(self._build_experiment(name, training, exp, summary))
        for name, exp in mlflow_exps.items():
            if name not in latest:
                all_experiments.append(self._build_experiment(name, None, exp, summaries.get(exp.experiment_id)))

        if search:
            search_lower = search.lower()
            filtered_list = [
                exp for exp in all_experiments
                if search_lower in exp["experiment_name"].lower()
            ]
        else:
            filtered_list = all_experiments

        total_items = len(filtered_list)
        total_pages = math.ceil(total_items / pageSize) if pageSize > 0 else 0

        # Ensure page is within bounds
        current_page = max(0, min(page, total_pages - 1)) if total_pages > 0 else 0
        start_index = current_page * pageSize
        end_index = start_index + pageSize

        metadata = PaginatedMetaData(
            total_items=total_items,
            total_pages=total_pages,
            current_page=current_page,
            page_size=pageSize,
            has_next=current_page < total_pages - 1,
            has_prev=current_page > 0
        )

        return PaginatedResult(
            results=filtered_list[start_index:end_index],
            pagination=metadata
        )

    async def get_experiment(self, experiment_name: str) -> Experiment:
        training = get_store().latest(experiment_name)
        mlflow_exp = get_active_experiment(experiment_name) if get_settings().mlflow_enabled else None

        if mlflow_exp is None and training is None:
            # Neither in MLflow nor in the local store: 404
            raise KeyError()

        summary = None
        if mlflow_exp is not None:
            summary = self._summarize_runs([mlflow_exp.experiment_id]).get(mlflow_exp.experiment_id)
        return self._build_experiment(experiment_name, training, mlflow_exp, summary)

    async def get_training_logs(self, experiment_name: str, tail_lines: int) -> TrainingLogs:
        """Logs of the training pod of an experiment (the active training, or else the latest one).
        Raises KeyError if the experiment does not exist. Non-404 Kubernetes errors are re-raised."""
        store = get_store()
        active = store.active(experiment_name)
        training = active[0] if active else store.latest(experiment_name)

        def result(status: str, available: bool, reason: Optional[str], logs: str = "",
                   training_id: Optional[str] = None, job_name: Optional[str] = None,
                   pod_name: Optional[str] = None, pod_phase: Optional[str] = None) -> TrainingLogs:
            return TrainingLogs(
                experiment_name=experiment_name,
                training_id=training_id,
                job_name=job_name,
                pod_name=pod_name,
                status=status,
                pod_phase=pod_phase,
                logs=logs,
                available=available,
                reason=reason,
            )

        if training is None:
            mlflow_exp = get_active_experiment(experiment_name) if get_settings().mlflow_enabled else None
            if mlflow_exp is None:
                raise KeyError()
            summary = self._summarize_runs([mlflow_exp.experiment_id]).get(mlflow_exp.experiment_id)
            status, _ = self._resolve_status(None, summary)
            return result(status, False, "This model was not trained by dflow, or its training is no longer kept.")

        training_id = training["training_id"]
        job_name = training.get("job_name") or job_name_for(training_id)
        status = training["status"]
        ids = {"training_id": training_id, "job_name": job_name}

        if status in ("queued", "launching_job"):
            return result(status, False, "The training has not started yet.", **ids)

        no_pod = "The training pod no longer exists; its logs are not kept."
        try:
            pods = await asyncio.to_thread(_job_pods, job_name)
        except ApiException as e:
            if e.status != 404:
                raise
            pods = []
        if not pods:
            return result(status, False, no_pod, **ids)

        pod = pods[0]
        pod_name = pod.metadata.name
        pod_phase = pod.status.phase if pod.status else None
        ids.update(pod_name=pod_name, pod_phase=pod_phase)

        def read_logs() -> str:
            return core_api().read_namespaced_pod_log(
                pod_name,
                get_settings().k8s_namespace,
                container=CONTAINER_NAME,
                tail_lines=tail_lines,
                timestamps=False,
            )

        try:
            logs = await asyncio.to_thread(read_logs)
        except ApiException as e:
            if e.status == 400:
                waiting = await asyncio.to_thread(get_pod_waiting_reason, job_name)
                reason = "The training pod is starting." + (f" ({waiting})" if waiting else "")
                return result(status, False, reason, **ids)
            if e.status == 404:
                return result(status, False, no_pod, **ids)
            raise

        return result(status, True, None, logs or "", **ids)

    # ---- commands ------------------------------------------------------------

    async def create_experiment(self,
                                background_tasks: BackgroundTasks,
                                csv_file: UploadFile,
                                model_type: str,
                                experiment_name: str,
                                model_name: str,
                                config_dict: dict[str, Any],
                                dependencies_list: List[Any]
    ) -> Experiment:
        """Registers a training and launches its Kubernetes Job in the background.
        Raises ActiveTrainingExists if the experiment already has a training in progress."""
        store = get_store()
        file_content = await csv_file.read()
        training_id = str(uuid.uuid4())[:8]
        filename = str(csv_file.filename if csv_file.filename is not None else "csvfile")

        # 1. Reserve the training first: the database rejects a second active training
        #    for the same experiment, before anything is uploaded.
        training = store.insert({
            "training_id": training_id,
            "experiment_name": experiment_name,
            "status": "queued",
            "model_type": model_type,
            "model_name": model_name,
            "config": config_dict,
            "dependencies": dependencies_list,
            "csv_filename": filename,
            "job_name": job_name_for(training_id),
        })

        # 2. Upload the dataset to MinIO
        try:
            minio_path = upload_csv_to_minio(file_content, training_id, filename)
            store.update(training_id, minio_path=minio_path)
        except Exception:
            store.delete(training_id)
            raise

        # 3. A training cannot log into a soft-deleted MLflow experiment with the same name
        if get_settings().mlflow_enabled:
            try:
                prepare_experiment_for_training(experiment_name)
            except Exception as e:
                print(f"[{training_id}] Could not prepare MLflow experiment '{experiment_name}': {e}")

        # 4. Delegate the execution to Kubernetes as a background task
        background_tasks.add_task(
            train_model_job,
            training_id=training_id,
            model_type=model_type,
            config=config_dict,
            minio_path=minio_path,
            experiment_name=experiment_name,
            model_name=model_name
        )

        return self._build_experiment(experiment_name, store.get(training_id) or training, None, None)

    def delete_experiment(self, experiment_name: str, mlflowb: bool) -> bool:
        """Deletes the experiment (local rows, MLflow and MinIO datasets), stopping the Job of any training in progress.

        The training CSVs of every training of the experiment are deleted from MinIO (a training
        without objects is not an error). If a Job cannot be deleted, the ApiException propagates;
        if MinIO fails, its error propagates. In both cases nothing else is deleted (rows and
        MLflow experiment are kept), so the call can be retried.
        """
        store = get_store()
        for training in store.active(experiment_name):
            job_name = training.get("job_name") or job_name_for(training["training_id"])
            # A Job that no longer exists counts as stopped (delete_training_job returns False).
            if delete_training_job(job_name):
                print(f"Kubernetes: job {job_name} of experiment '{experiment_name}' deleted.")

        for training in store.list_by_name(experiment_name):
            removed = delete_training_data(training["training_id"])
            print(f"MinIO: {removed} object(s) of training {training['training_id']} deleted.")

        found = False
        deleted = store.delete_by_name(experiment_name)
        if deleted:
            found = True
            print(f"Local DB: {deleted} training(s) of experiment '{experiment_name}' deleted.")

        if mlflowb:
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

        return found

    def cancel_training(self, experiment_name: str) -> Dict[str, Any]:
        """Stops the training in progress of an experiment and removes it.

        Deletes the Kubernetes Job (and its pods), kills and deletes the MLflow run of this
        training and its uploaded dataset. If the model has no other versions, the whole
        experiment is deleted; otherwise only this training is removed and the existing
        versions are left untouched. Raises NoActiveTraining if nothing is in progress.
        """
        store = get_store()

        active = store.active(experiment_name)
        if active:
            training_id = active[0]["training_id"]

            # Refresh first: the training may have just finished.
            try:
                reconcile_training(training_id)
            except Exception as e:
                print(f"[{training_id}] Could not refresh status before cancelling: {e}")

            # Compare-and-set to 'cancelled': from now on the launcher and the poller leave it alone.
            if not store.update(training_id, expected_status=ACTIVE_STATUSES, status="cancelled"):
                raise NoActiveTraining(experiment_name)
        else:
            # A previous cancellation whose cleanup failed: retry it.
            pending = [t for t in store.list_by_name(experiment_name) if t["status"] == "cancelled"]
            if not pending:
                raise NoActiveTraining(experiment_name)
            training_id = pending[0]["training_id"]

        training = store.get(training_id)
        if training is None:
            raise NoActiveTraining(experiment_name)

        result = cleanup_cancelled_training(training)
        if not result["cleanup_complete"]:
            outcome = "cleanup incomplete, it will be retried automatically"
        elif result["experiment_deleted"]:
            outcome = "experiment deleted because it had no versions"
        else:
            outcome = "the existing model versions were kept"

        return {
            "message": f"Training '{training_id}' of '{experiment_name}' cancelled; {outcome}",
            "experiment_name": experiment_name,
            "training_id": training_id,
            "status": "cancelled",
            "experiment_deleted": result["experiment_deleted"],
            "cleanup_complete": result["cleanup_complete"],
            "deleted": result["deleted"],
            "errors": result["errors"],
        }

    def _to_timestamp_int(self, val: Any) -> Optional[int]:
        """
        Safely converts ISO string, pd.Timestamp, datetime, or raw numeric epoch 
        to epoch milliseconds (int). Returns None on missing/invalid values.
        """
        if val is None:
            return None

        # Handle pandas NaT or NaN
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass

        # 1. Handle Pandas Timestamp or standard datetime object
        if isinstance(val, (pd.Timestamp, datetime.datetime)):
            return int(val.timestamp() * 1000)

        # 2. Handle raw numeric epoch timestamps
        if isinstance(val, (int, float)):
            val_int = int(val)
            # If already in nanoseconds (typical Pandas underlying int: > 1e18)
            if val_int > 1e16:
                return val_int // 1_000_000
            # If already in milliseconds (typical MLflow native int: ~1.7e12)
            elif val_int > 1e11:
                return val_int
            # If in seconds (~1.7e9)
            else:
                return val_int * 1000

        # 3. Handle ISO 8601 string (e.g., '2026-10-16T12:23:03.033000+00:00')
        if isinstance(val, str):
            cleaned = val.strip()
            if not cleaned:
                return None
            try:
                # Handle standard ISO strings
                dt = datetime.datetime.fromisoformat(cleaned)
                return int(dt.timestamp() * 1000)
            except ValueError:
                # Fallback for ISO strings with 'Z' instead of '+00:00'
                try:
                    dt = datetime.datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
                    return int(dt.timestamp() * 1000)
                except ValueError:
                    return None

        return None


    def parse_flattened_run(self, raw_data: dict[str, Any], source: str = "mlflow") -> Run:
        metrics: dict[str, float] = {}
        params: dict[str, Any] = {}
        tags: dict[str, str] = {}

        # Separate prefixed fields
        for key, value in raw_data.items():
            if key.startswith("metrics."):
                clean_key = key[len("metrics."):]
                try:
                    metrics[clean_key] = float(value)
                except (ValueError, TypeError):
                    pass
            elif key.startswith("params."):
                clean_key = key[len("params."):]
                params[clean_key] = value
            elif key.startswith("tags."):
                clean_key = key[len("tags."):]
                tags[clean_key] = str(value)

        return Run(
            run_id=str(raw_data.get("run_id", "")),
            source=source,
            status=str(raw_data.get("status", "UNKNOWN")),
            experiment_id=str(raw_data.get("experiment_id", "")),
            artifact_uri=raw_data.get("artifact_uri"),
            start_time=self._to_timestamp_int(raw_data.get("start_time")),
            end_time=self._to_timestamp_int(raw_data.get("end_time")),
            metrics=metrics,
            params=params,
            tags=tags,
        )

    def get_experiments_runs(self, experiment_name: str) -> Generator[Run, None, None]:
        mlflow_exp = get_active_experiment(experiment_name)
        has_local = get_store().latest(experiment_name) is not None

        if not mlflow_exp and not has_local:
            # if it's not either in the local database, return 404
            raise KeyError()

        if not mlflow_exp and has_local:
            return

        assert mlflow_exp is not None

        runs_df = mlflow.search_runs(
            experiment_ids=[mlflow_exp.experiment_id],
            order_by=["attribute.start_time DESC"]
        )

        if runs_df.empty:
            return  # Yield nothing if the dataframe is empty

        # Replace pandas NaNs and NaTs with standard Python None
        clean_df = runs_df.where(pd.notnull(runs_df), None)
        
        # Convert to a list of standard dictionaries
        run_records = cast(list[dict[str, Any]], clean_df.to_dict(orient="records"))

        for run_dict in run_records:
            yield self.parse_flattened_run(run_dict)

    def get_runs(self, experiment_name:str, page: int = 0, pageSize: int = 20, search: str = '') -> PaginatedResult[Run]:
        all_runs = list(self.get_experiments_runs(experiment_name))

        if search:
            search_lower = search.lower()
            filtered_runs = []
            
            for run in all_runs:
                run_id = run.get("run_id", "").lower()
                # MLflow usually stores the human-readable name in this tag
                run_name = run.get("tags", {}).get("mlflow.runName", "").lower()
                
                # If the query matches the ID or the Name, include it
                if search_lower in run_id or search_lower in run_name:
                    filtered_runs.append(run)
        else:
            filtered_runs = all_runs

        total_items = len(filtered_runs)
        total_pages = math.ceil(total_items / pageSize) if pageSize > 0 else 0
        
        current_page = max(0, min(page, total_pages - 1)) if total_pages > 0 else 0
        start = current_page * pageSize
        end = start + pageSize

        metadata = PaginatedMetaData(
            total_items=total_items,
            total_pages=total_pages,
            current_page=current_page,
            page_size=pageSize,
            has_next=current_page < total_pages - 1,
            has_prev= current_page>0
        )

        return PaginatedResult(
            results=filtered_runs[start:end],
            pagination=metadata
        )