from types import SimpleNamespace

import pytest
from kubernetes.client.rest import ApiException

import src.service as service_module
from tests.conftest import FakeCoreApi, make_pod, make_training

URL = "/api/mlops/experiments/my-exp/logs"
NO_POD = "The training pod no longer exists; its logs are not kept."
NOT_STARTED = "The training has not started yet."
STARTING = "The training pod is starting."
NOT_DFLOW = "This model was not trained by dflow, or its training is no longer kept."


def test_active_training_returns_logs(client, set_store, k8s):
    training = make_training("training", "abc12345", "dflow-train-job-abc12345")
    set_store(active=training, latest=training)
    k8s.pods = [make_pod("pod-xyz", "Running")]
    k8s.core = FakeCoreApi(logs="line1\nline2")

    r = client.get(URL, params={"tail_lines": 50})

    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["logs"] == "line1\nline2"
    assert body["reason"] is None
    assert body["pod_name"] == "pod-xyz"
    assert body["pod_phase"] == "Running"
    assert body["training_id"] == "abc12345"
    assert body["job_name"] == "dflow-train-job-abc12345"
    assert body["status"] == "training"
    assert body["experiment_name"] == "my-exp"
    assert k8s.job_pods_calls == ["dflow-train-job-abc12345"]
    ((args, kwargs),) = k8s.core.calls
    assert args[0] == "pod-xyz"
    assert args[1] == "test-namespace"
    assert kwargs["container"] == "trainer"
    assert kwargs["tail_lines"] == 50
    assert kwargs["timestamps"] is False


def test_default_tail_lines_is_500(client, set_store, k8s):
    set_store(active=make_training())
    k8s.core = FakeCoreApi(logs="x")
    assert client.get(URL).status_code == 200
    assert k8s.core.calls[0][1]["tail_lines"] == 500


@pytest.mark.parametrize("value", [0, 5001])
def test_tail_lines_out_of_range_is_422(client, set_store, k8s, value):
    set_store(active=make_training())
    assert client.get(URL, params={"tail_lines": value}).status_code == 422
    assert k8s.core.calls == []


def test_active_training_preferred_over_latest(client, set_store, k8s):
    set_store(active=make_training("training", "active01"), latest=make_training("completed", "latest01"))
    k8s.core = FakeCoreApi(logs="x")

    body = client.get(URL).json()

    assert body["training_id"] == "active01"
    assert body["job_name"] == "dflow-train-job-active01"
    assert k8s.job_pods_calls == ["dflow-train-job-active01"]


def test_latest_used_when_no_active(client, set_store, k8s):
    set_store(active=None, latest=make_training("completed", "latest01"))
    k8s.core = FakeCoreApi(logs="done")

    body = client.get(URL).json()

    assert body["training_id"] == "latest01"
    assert body["status"] == "completed"
    assert body["logs"] == "done"


@pytest.mark.parametrize("status", ["queued", "launching_job"])
def test_not_started_does_not_call_kubernetes(client, set_store, k8s, status):
    set_store(active=make_training(status, "abc12345"))

    r = client.get(URL)

    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == NOT_STARTED
    assert body["status"] == status
    assert body["logs"] == ""
    assert body["training_id"] == "abc12345"
    assert body["pod_name"] is None
    assert k8s.job_pods_calls == []
    assert k8s.core.calls == []


def test_no_pods(client, set_store, k8s):
    set_store(latest=make_training("completed"))
    k8s.pods = []

    body = client.get(URL).json()

    assert body["available"] is False
    assert body["reason"] == NO_POD
    assert body["pod_name"] is None
    assert k8s.core.calls == []


def test_job_pods_404_means_no_pod(client, set_store, k8s):
    set_store(latest=make_training("completed"))
    k8s.pods_error = ApiException(status=404, reason="Not Found")

    r = client.get(URL)

    assert r.status_code == 200
    assert r.json()["available"] is False
    assert r.json()["reason"] == NO_POD


def test_read_400_pod_starting_without_waiting_reason(client, set_store, k8s):
    set_store(active=make_training())
    k8s.core = FakeCoreApi(error=ApiException(status=400, reason="Bad Request"))
    k8s.waiting_reason = None

    body = client.get(URL).json()

    assert body["available"] is False
    assert body["reason"] == STARTING
    assert body["pod_name"] == "pod-1"
    assert k8s.waiting_calls == ["dflow-train-job-abc12345"]


def test_read_400_pod_starting_with_waiting_reason(client, set_store, k8s):
    set_store(active=make_training())
    k8s.core = FakeCoreApi(error=ApiException(status=400, reason="Bad Request"))
    k8s.waiting_reason = "ImagePullBackOff: back-off pulling image"

    body = client.get(URL).json()

    assert body["available"] is False
    assert body["reason"] == f"{STARTING} (ImagePullBackOff: back-off pulling image)"


def test_read_404_means_no_pod(client, set_store, k8s):
    set_store(active=make_training())
    k8s.core = FakeCoreApi(error=ApiException(status=404, reason="Not Found"))

    body = client.get(URL).json()

    assert body["available"] is False
    assert body["reason"] == NO_POD
    assert body["pod_name"] == "pod-1"


@pytest.mark.parametrize("status,reason", [(500, "Internal Server Error"), (403, "Forbidden")])
def test_job_pods_other_error_is_502(client, set_store, k8s, status, reason):
    set_store(active=make_training())
    k8s.pods_error = ApiException(status=status, reason=reason)

    r = client.get(URL)

    assert r.status_code == 502
    assert str(status) in r.json()["detail"]
    assert reason in r.json()["detail"]


@pytest.mark.parametrize("status,reason", [(500, "Internal Server Error"), (403, "Forbidden")])
def test_read_other_error_is_502(client, set_store, k8s, status, reason):
    set_store(active=make_training())
    k8s.core = FakeCoreApi(error=ApiException(status=status, reason=reason))

    r = client.get(URL)

    assert r.status_code == 502
    assert str(status) in r.json()["detail"]
    assert reason in r.json()["detail"]


def test_no_local_training_but_in_mlflow(client, set_store, k8s, monkeypatch):
    set_store(active=None, latest=None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.invalid")
    monkeypatch.setattr(service_module, "get_active_experiment", lambda name: SimpleNamespace(experiment_id="1"))
    monkeypatch.setattr(
        service_module.DFlowService,
        "_summarize_runs",
        staticmethod(lambda ids: {"1": service_module._RunSummary(latest_status="FINISHED", has_finished=True)}),
    )

    r = client.get(URL)

    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == NOT_DFLOW
    assert body["status"] == "completed"
    assert body["training_id"] is None
    assert k8s.job_pods_calls == []
    assert k8s.core.calls == []


def test_no_local_training_and_not_in_mlflow(client, set_store, k8s, monkeypatch):
    set_store(active=None, latest=None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.invalid")
    monkeypatch.setattr(service_module, "get_active_experiment", lambda name: None)

    r = client.get(URL)

    assert r.status_code == 404
    assert r.json()["detail"] == "There is no experiment with name my-exp"


def test_no_local_training_and_mlflow_disabled(client, set_store, k8s, monkeypatch):
    set_store(active=None, latest=None)

    def must_not_be_called(name):
        raise AssertionError("MLflow must not be queried when disabled")

    monkeypatch.setattr(service_module, "get_active_experiment", must_not_be_called)

    r = client.get(URL)

    assert r.status_code == 404
    assert r.json()["detail"] == "There is no experiment with name my-exp"


@pytest.mark.parametrize("empty", ["", None])
def test_empty_logs_are_available(client, set_store, k8s, empty):
    set_store(active=make_training())
    k8s.core = FakeCoreApi(logs=empty)

    body = client.get(URL).json()

    assert body["available"] is True
    assert body["logs"] == ""
    assert body["reason"] is None
