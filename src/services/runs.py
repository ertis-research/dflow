"""MLflow runs of an experiment."""
import datetime
from typing import Any, Generator, Optional, cast

import mlflow
import pandas as pd

from src.db.store import get_store
from src.integrations.mlflow_tracking import get_active_experiment
from src.schemas import PaginatedResult, Run
from src.services.pagination import paginate


def to_timestamp_ms(val: Any) -> Optional[int]:
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


def parse_flattened_run(raw_data: dict[str, Any], source: str = "mlflow") -> Run:
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
        start_time=to_timestamp_ms(raw_data.get("start_time")),
        end_time=to_timestamp_ms(raw_data.get("end_time")),
        metrics=metrics,
        params=params,
        tags=tags,
    )


def iter_experiment_runs(experiment_name: str) -> Generator[Run, None, None]:
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
        yield parse_flattened_run(run_dict)


def get_runs(experiment_name: str, page: int = 0, pageSize: int = 20, search: str = '') -> PaginatedResult[Run]:
    all_runs = list(iter_experiment_runs(experiment_name))

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

    return paginate(filtered_runs, page, pageSize)
