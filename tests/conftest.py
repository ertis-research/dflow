"""Shared fixtures. Tests never touch the real .env or external services."""
import os
from types import SimpleNamespace

# Dummy settings, set before src is imported. load_dotenv() does not override
# variables that already exist, so the real .env cannot leak into the tests.
_DUMMY_ENV = {
    "MINIO_URL": "minio.invalid:9000",
    "MINIO_ACCESS_KEY": "test-access-key",
    "MINIO_SECRET_KEY": "test-secret-key",
    "MINIO_SECURE": "false",
    "BUCKET_NAME": "test-bucket",
    "KUBERNETES_NAMESPACE": "test-namespace",
    "MLFLOW_TRACKING_URI": "",  # empty: MLflow disabled unless a test enables it
    "DB_BACKEND": "sqlite",
    "SQLITE_PATH": ":memory:",
}
os.environ.update(_DUMMY_ENV)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.config import get_settings  # noqa: E402

get_settings.cache_clear()

import src.main  # noqa: E402
import src.services.experiments as experiments_module  # noqa: E402
import src.services.trainings as trainings_module  # noqa: E402


@pytest.fixture(autouse=True)
def _settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    # No "with" block: the lifespan (init_store, ensure_bucket, poller) must not run.
    return TestClient(src.main.app)


class FakeStore:
    def __init__(self, active=None, latest=None, history=None):
        self._history = history or []
        self._active = active
        self._latest = latest
        self.deleted_names = []

    def delete_by_name(self, experiment_name):
        self.deleted_names.append(experiment_name)
        return 1 if (self._active or self._latest) else 0

    def list_by_name(self, experiment_name):
        trainings = list(self._history)
        for t in (self._active, self._latest):
            if t and t not in trainings:
                trainings.append(t)
        return trainings

    def active(self, experiment_name):
        return [self._active] if self._active else []

    def latest(self, experiment_name):
        return self._latest


def make_training(status="training", training_id="abc12345", job_name=None):
    return {"training_id": training_id, "status": status, "job_name": job_name or f"dflow-train-job-{training_id}"}


def make_pod(name="pod-1", phase="Running", created="2026-01-01T00:00:00Z"):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, creation_timestamp=created),
        status=SimpleNamespace(phase=phase),
    )


class FakeCoreApi:
    def __init__(self, logs="", error=None):
        self.logs = logs
        self.error = error
        self.calls = []

    def read_namespaced_pod_log(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.logs


@pytest.fixture
def k8s(monkeypatch):
    """Patches the Kubernetes helpers as used in src.services.trainings. Returns a configurable state object."""
    state = SimpleNamespace(
        pods=[make_pod()],
        pods_error=None,
        job_pods_calls=[],
        core=FakeCoreApi(),
        waiting_reason=None,
        waiting_calls=[],
    )

    def fake_job_pods(job_name):
        state.job_pods_calls.append(job_name)
        if state.pods_error is not None:
            raise state.pods_error
        return state.pods

    def fake_waiting(job_name):
        state.waiting_calls.append(job_name)
        return state.waiting_reason

    monkeypatch.setattr(trainings_module, "job_pods", fake_job_pods)
    monkeypatch.setattr(trainings_module, "core_api", lambda: state.core)
    monkeypatch.setattr(trainings_module, "get_pod_waiting_reason", fake_waiting)
    return state


@pytest.fixture
def set_store(monkeypatch):
    def _set(active=None, latest=None, history=None):
        store = FakeStore(active=active, latest=latest, history=history)
        monkeypatch.setattr(experiments_module, "get_store", lambda: store)
        monkeypatch.setattr(trainings_module, "get_store", lambda: store)
        return store
    return _set
