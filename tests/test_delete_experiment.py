from types import SimpleNamespace

import pytest
from kubernetes.client.rest import ApiException
from minio.error import S3Error

import src.api.experiments as api_module
import src.services.experiments as experiments_module
from tests.conftest import make_training

URL = "/api/mlops/experiments/my-exp"


@pytest.fixture(autouse=True)
def minio_delete(monkeypatch):
    """Patches delete_training_data as used in src.services.experiments: these tests never touch MinIO."""
    state = SimpleNamespace(calls=[], error=None)

    def fake_delete(training_id):
        state.calls.append(training_id)
        if state.error is not None:
            raise state.error
        return 0  # no objects is not an error

    monkeypatch.setattr(experiments_module, "delete_training_data", fake_delete)
    return state


def _patch_delete_job(monkeypatch, result=True, error=None):
    calls = []

    def fake_delete(job_name):
        calls.append(job_name)
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(experiments_module, "delete_training_job", fake_delete)
    return calls


def test_active_training_job_is_deleted_then_rows(client, set_store, monkeypatch):
    store = set_store(active=make_training(job_name="my-job"))
    calls = _patch_delete_job(monkeypatch)

    response = client.delete(URL)

    assert response.status_code == 200
    assert calls == ["my-job"]
    assert store.deleted_names == ["my-exp"]


def test_missing_job_counts_as_ok(client, set_store, monkeypatch):
    store = set_store(active=make_training(job_name="my-job"))
    _patch_delete_job(monkeypatch, result=False)

    assert client.delete(URL).status_code == 200
    assert store.deleted_names == ["my-exp"]


def test_job_name_falls_back_to_training_id(client, set_store, monkeypatch):
    training = make_training(training_id="ABC123")
    training["job_name"] = None
    set_store(active=training)
    calls = _patch_delete_job(monkeypatch)

    assert client.delete(URL).status_code == 200
    assert calls == ["dflow-train-job-abc123"]


def test_no_active_training_does_not_touch_kubernetes(client, set_store, monkeypatch):
    store = set_store(latest=make_training(status="completed"))
    calls = _patch_delete_job(monkeypatch)

    response = client.delete(URL)

    assert response.status_code == 200
    assert calls == []
    assert store.deleted_names == ["my-exp"]


def test_kubernetes_error_returns_502_and_keeps_rows(client, set_store, monkeypatch):
    store = set_store(active=make_training(job_name="my-job"))
    _patch_delete_job(monkeypatch, error=ApiException(status=500, reason="Internal Server Error"))

    response = client.delete(URL)

    assert response.status_code == 502
    assert "500 Internal Server Error" in response.json()["detail"]
    assert store.deleted_names == []


def test_unknown_experiment_returns_404(client, set_store, monkeypatch):
    set_store()
    calls = _patch_delete_job(monkeypatch)

    assert client.delete(URL).status_code == 404
    assert calls == []


def test_minio_data_of_all_trainings_is_deleted(client, set_store, monkeypatch, minio_delete):
    old1 = make_training(status="completed", training_id="old1")
    old2 = make_training(status="failed", training_id="old2")
    active = make_training(training_id="act1", job_name="my-job")
    store = set_store(active=active, history=[old1, old2])
    _patch_delete_job(monkeypatch)

    response = client.delete(URL)

    assert response.status_code == 200
    assert sorted(minio_delete.calls) == ["act1", "old1", "old2"]
    assert store.deleted_names == ["my-exp"]


def test_minio_error_returns_502_and_keeps_rows_and_mlflow(client, set_store, monkeypatch, minio_delete):
    store = set_store(active=make_training(job_name="my-job"))
    _patch_delete_job(monkeypatch)
    minio_delete.error = S3Error("AccessDenied", "Access Denied", "r", "req", "host", None)
    mlflow_calls = []
    fake_mlflow = SimpleNamespace(
        get_experiment_by_name=lambda name: mlflow_calls.append(name),
        delete_experiment=lambda eid: mlflow_calls.append(eid),
    )
    monkeypatch.setattr(experiments_module, "mlflow", fake_mlflow)
    monkeypatch.setattr(api_module, "mlflow_enabled", lambda: True)

    response = client.delete(URL)

    assert response.status_code == 502
    assert response.json()["detail"].startswith("MinIO error while deleting training data:")
    assert store.deleted_names == []
    assert mlflow_calls == []


def test_unknown_experiment_does_not_touch_minio(client, set_store, monkeypatch, minio_delete):
    set_store()
    _patch_delete_job(monkeypatch)

    assert client.delete(URL).status_code == 404
    assert minio_delete.calls == []
