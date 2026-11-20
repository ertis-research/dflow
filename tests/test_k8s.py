from src.integrations.k8s import job_name_for


def test_job_name_has_dflow_prefix_and_is_lowercase():
    assert job_name_for("AbC123") == "dflow-train-job-abc123"
