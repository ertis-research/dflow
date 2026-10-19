from typing import Any, Generic, List, Optional, TypeVar, TypedDict


class Experiment(TypedDict):
    experiment_name: str
    source: str
    status: Optional[str]
    internal_id: Optional[str]
    mlflow_id: Optional[str]
    artifact_location: Optional[str]
    model: Optional[str]
    model_name: Optional[str]
    config: Optional[dict[str, Any]]
    dependencies: Optional[List[Any]]
    csv_filename: Optional[str]

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