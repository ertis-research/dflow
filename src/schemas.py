from typing import Any, Generic, List, Literal, Optional, TypeVar, TypedDict


# Closed vocabulary for the training status of an experiment (model).
#   queued        -> request accepted and dataset stored; the Kubernetes Job is not requested yet
#   launching_job -> the Kubernetes Job is being created
#   training      -> the Job exists and has not finished (includes a Pending pod)
#   completed     -> the Job succeeded and the run is registered in MLflow
#   failed        -> the training failed; see `status_reason`
#   cancelled     -> only returned by the cancel endpoint (the record is then removed)
TrainingStatus = Literal["queued", "launching_job", "training", "completed", "failed", "cancelled"]

ALL_STATUSES = ("queued", "launching_job", "training", "completed", "failed", "cancelled")
ACTIVE_STATUSES = ("queued", "launching_job", "training")


class Experiment(TypedDict):
    experiment_name: str
    source: str
    status: Optional[str]
    status_reason: Optional[str]
    internal_id: Optional[str]
    mlflow_id: Optional[str]
    artifact_location: Optional[str]
    model: Optional[str]
    model_name: Optional[str]
    config: Optional[dict[str, Any]]
    dependencies: Optional[List[Any]]
    csv_filename: Optional[str]
    created_at: Optional[int]

class Run(TypedDict):
    run_id: str
    source: str
    status: str
    experiment_id: str
    artifact_uri: Optional[str]
    start_time: Optional[int]
    end_time: Optional[int]

    metrics: dict[str, float]
    params: dict[str, Any]
    tags: dict[str, str]


class TrainingLogs(TypedDict):
    experiment_name: str
    training_id: Optional[str]
    job_name: Optional[str]
    pod_name: Optional[str]
    status: str
    pod_phase: Optional[str]
    logs: str
    available: bool
    reason: Optional[str]


class PaginatedMetaData(TypedDict):
    total_items: int
    total_pages: int
    current_page: int
    page_size: int
    has_next: bool
    has_prev: bool

T = TypeVar("T")

class PaginatedResult(TypedDict, Generic[T]):
    results: List[T]
    pagination: PaginatedMetaData
