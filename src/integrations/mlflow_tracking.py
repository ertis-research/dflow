"""Helpers to relate dflow trainings with MLflow experiments and runs."""
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.entities import Experiment as MlflowExperiment
from mlflow.entities import Run as MlflowRun
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

# Tag set by the training pod on the run it registers.
TRAINING_ID_TAG = "dflow.training_id"

ACTIVE_RUN_STATUSES = ("RUNNING", "SCHEDULED")


def get_active_experiment(experiment_name: str) -> Optional[MlflowExperiment]:
    """MLflow experiment by name, ignoring soft-deleted ones."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None or exp.lifecycle_stage != "active":
        return None
    return exp


def find_training_runs(experiment_id: str, training: Dict[str, Any]) -> List[MlflowRun]:
    """Runs created by one dflow training.

    Runs are matched by the `dflow.training_id` tag. Runs without that tag (created before the tag
    existed, or still being logged) are matched if they started after the training was created.
    At most one training per experiment can be active, so this cannot pick runs of another training.
    """
    client = MlflowClient()
    tagged = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`{TRAINING_ID_TAG}` = '{training['training_id']}'",
    )
    if tagged:
        return list(tagged)

    created_at = int(training["created_at"])
    candidates = client.search_runs(
        [experiment_id],
        filter_string=f"attributes.start_time >= {created_at}",
    )
    return [run for run in candidates if TRAINING_ID_TAG not in run.data.tags]


def find_finished_training_run(training: Dict[str, Any]) -> Optional[MlflowRun]:
    exp = get_active_experiment(training["experiment_name"])
    if exp is None:
        return None
    for run in find_training_runs(exp.experiment_id, training):
        if run.info.status == "FINISHED":
            return run
    return None


def has_finished_runs(experiment_id: str, exclude_run_ids: Optional[List[str]] = None) -> bool:
    """True if the experiment has at least one FINISHED (non-deleted) run, i.e. a model version."""
    exclude = set(exclude_run_ids or [])
    runs = MlflowClient().search_runs(
        [experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    return any(run.info.run_id not in exclude for run in runs)


def discard_run(run: MlflowRun) -> List[str]:
    """Marks a run as KILLED if it is still running, deletes the model versions registered from it
    and deletes the run. Returns the deleted model versions as 'name/version'."""
    client = MlflowClient()
    run_id = run.info.run_id
    if run.info.status in ACTIVE_RUN_STATUSES:
        client.set_terminated(run_id, "KILLED")

    deleted_versions: List[str] = []
    for version in client.search_model_versions(f"run_id='{run_id}'"):
        client.delete_model_version(version.name, version.version)
        deleted_versions.append(f"{version.name}/{version.version}")

    client.delete_run(run_id)
    return deleted_versions


def prepare_experiment_for_training(experiment_name: str) -> None:
    """MLflow only soft-deletes experiments, and a training cannot log into a deleted experiment.
    If the name belongs to a deleted experiment, restore it and delete its old runs, so the new
    training starts clean (the user had already deleted that model)."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None or exp.lifecycle_stage != "deleted":
        return
    client.restore_experiment(exp.experiment_id)
    for run in client.search_runs([exp.experiment_id], run_view_type=ViewType.ACTIVE_ONLY):
        client.delete_run(run.info.run_id)
    print(f"MLflow: restored deleted experiment '{experiment_name}' and cleared its old runs")
