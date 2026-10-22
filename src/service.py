import datetime
import math
from typing import Any, Generator, List, Optional, cast
import uuid
import pandas as pd
from fastapi import BackgroundTasks, UploadFile
import mlflow

from src.k8s_utils import train_model_job
from src.minio_utils import upload_csv_to_minio
from src.schemas import Experiment, PaginatedResult, PaginatedMetaData, Run
from src.database import fake_database


class DFlowService:
    def __init__(self) -> None:
        pass

    def get_local_experiments(self) -> Generator[Experiment, None, None]:
        for eid, data in fake_database.items():
            name = data.get("experiment_name", f"local-{eid}")
            yield Experiment(
                experiment_name=name,
                source="local",
                status=data.get("status"),
                internal_id=eid,
                mlflow_id=None,  # Local doesn't have this yet
                artifact_location=None,
                model=data.get("model"),
                model_name=data.get("model_name"),
                config=data.get("config"),
                dependencies=data.get("dependencies", []),
                csv_filename=data.get("csv_filename")
            )

    def get_mlflow_experiments(self, locals: List[str]) -> Generator[Experiment, None, None]:
        try:
            mlflow_exps = mlflow.search_experiments()
            for exp in mlflow_exps:
                if exp.name == "Default":
                    continue

                if exp.name not in locals:
                    yield Experiment(
                        experiment_name=exp.name,
                        source="mlflow",
                        status="completed", # MLflow runs are usually completed/logged
                        internal_id=None,
                        mlflow_id=exp.experiment_id,
                        artifact_location=exp.artifact_location,
                        model=None,         # MLflow API doesn't expose this at this level
                        model_name=None,
                        config=None,
                        dependencies=[],
                        csv_filename=None
                    )
            
        except Exception as e:
            print(f"Error reading from MLflow: {str(e)}")

    def get_all_experiments(self, mlflow: bool, page: int = 0, pageSize: int = 20, search: str = '') -> PaginatedResult[Experiment]:
        locals: List[Experiment] = list(self.get_local_experiments())
        localsIds = []
        for exp in locals:
            localsIds.append(exp["experiment_name"])
        mlflows: List[Experiment] = list(self.get_mlflow_experiments(localsIds)) if mlflow else []

        all_experiments = locals + mlflows

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
        print(1)
        mlflow_exp = mlflow.get_experiment_by_name(experiment_name)
        print(2)
        exp = [(k, v) for k, v in fake_database.items() if v.get("experiment_name") == experiment_name]
        print(3)

        if not mlflow_exp and len(exp)==0:
            # if it's not either in the local database, return 404
            raise KeyError()
        print(4)
        if not mlflow_exp and len(exp)>0:
            res_exp = exp[0]
            return Experiment(
                experiment_name=experiment_name,
                source="local",
                status=res_exp[1].get("status"),
                internal_id=res_exp[0],
                mlflow_id=None,  # Local doesn't have this yet
                artifact_location=None,
                model=res_exp[1].get("model"),
                model_name=res_exp[1].get("model_name"),
                config=res_exp[1].get("config"),
                dependencies=res_exp[1].get("dependencies", []),
                csv_filename=res_exp[1].get("csv_filename")
            )

        assert mlflow_exp is not None
        print(5)
        return Experiment(
            experiment_name=mlflow_exp.name,
            source="mlflow",
            status="completed", # MLflow runs are usually completed/logged
            internal_id=None,
            mlflow_id=mlflow_exp.experiment_id,
            artifact_location=mlflow_exp.artifact_location,
            model=None,         # MLflow API doesn't expose this at this level
            model_name=None,
            config=None,
            dependencies=[],
            csv_filename=None
        )
        

    async def create_experiment(self, 
                                background_tasks: BackgroundTasks, 
                                csv_file: UploadFile, 
                                model_type: str, 
                                experiment_name: str, 
                                model_name: str,
                                config_dict: dict[str, Any],
                                dependencies_list: List[Any]
    ) -> Experiment:
        file_content = await csv_file.read()
        experiment_id = str(uuid.uuid4())[:8]
        filename = str(csv_file.filename if csv_file.filename is not None else "csvfile")
        
        # 2. Delegate the file upload to MinIO
        minio_path = upload_csv_to_minio(file_content, experiment_id, filename)

        experiment = Experiment(
            experiment_name=experiment_name,
            source="local",
            status="pending_training", # MLflow runs are usually completed/logged
            internal_id=experiment_id,
            mlflow_id=None,
            artifact_location=None,
            model=model_type,         # MLflow API doesn't expose this at this level
            model_name=model_name,
            config=config_dict,
            dependencies=dependencies_list,
            csv_filename=filename
        )     

        fake_database[experiment_id] = experiment;  

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
        
        return  fake_database[experiment_id]

    def delete_experiment(self, experiment_name: str, mlflowb: bool) -> bool:
        found = False
        keys_to_delete = [k for k, v in fake_database.items() if v.get("experiment_name") == experiment_name]
        for k in keys_to_delete:
            del fake_database[k]
            found = True
            print(f"Local DB: Experiment '{experiment_name}' (ID: {k}) deleted.")

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
        mlflow_exp = mlflow.get_experiment_by_name(experiment_name)
        exp = [k for k, v in fake_database.items() if v.get("experiment_name") == experiment_name]

        if not mlflow_exp and len(exp)==0:
            # if it's not either in the local database, return 404
            raise KeyError()

        if not mlflow_exp and len(exp)>0:
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