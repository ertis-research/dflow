"""
Centralised configuration for dflow.

Every setting comes from environment variables (a local ``.env`` file in
development, a Secret/ConfigMap in Kubernetes). Required variables have no
default: if one is missing, dflow refuses to start and names the variable.
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _optional(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, "").strip()
    return value if value else default


def _int(name: str, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    raw = _optional(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"Environment variable {name} must be an integer, got '{raw}'")
    if minimum is not None and value < minimum:
        raise ConfigError(f"Environment variable {name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"Environment variable {name} must be <= {maximum}, got {value}")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = _optional(name)
    if raw is None:
        return default
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"Environment variable {name} must be true/false, got '{raw}'")


@dataclass(frozen=True)
class Settings:
    # MinIO
    minio_url: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    bucket_name: str

    # Kubernetes
    kube_config_path: str
    k8s_namespace: str
    training_image: str
    training_job_ttl_seconds: int
    training_job_backoff_limit: int

    # MLflow (optional: without it dflow only tracks local trainings)
    mlflow_tracking_uri: Optional[str]

    # Status tracking
    status_poll_interval_seconds: int
    launch_timeout_seconds: int

    # Storage
    db_backend: str
    sqlite_path: str
    postgres_host: Optional[str]
    postgres_port: int
    postgres_db: Optional[str]
    postgres_user: Optional[str]
    postgres_password: Optional[str]

    @property
    def mlflow_enabled(self) -> bool:
        return bool(self.mlflow_tracking_uri)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    db_backend = (_optional("DB_BACKEND", "sqlite") or "sqlite").lower()
    if db_backend not in ("sqlite", "postgres"):
        raise ConfigError(f"DB_BACKEND must be 'sqlite' or 'postgres', got '{db_backend}'")

    settings = Settings(
        minio_url=_required("MINIO_URL"),
        minio_access_key=_required("MINIO_ACCESS_KEY"),
        minio_secret_key=_required("MINIO_SECRET_KEY"),
        minio_secure=_bool("MINIO_SECURE", True),
        bucket_name=_required("BUCKET_NAME"),
        kube_config_path=os.path.expanduser(_optional("KUBERNETES_CONFIG_PATH", "~/.kube/config") or ""),
        k8s_namespace=_required("KUBERNETES_NAMESPACE"),
        training_image=_optional("TRAINING_IMAGE", "ludwigai/ludwig:latest") or "",
        training_job_ttl_seconds=_int("TRAINING_JOB_TTL_SECONDS", 60, minimum=0),
        training_job_backoff_limit=_int("TRAINING_JOB_BACKOFF_LIMIT", 1, minimum=0),
        mlflow_tracking_uri=_optional("MLFLOW_TRACKING_URI"),
        status_poll_interval_seconds=_int("STATUS_POLL_INTERVAL_SECONDS", 10, minimum=1),
        launch_timeout_seconds=_int("LAUNCH_TIMEOUT_SECONDS", 300, minimum=1),
        db_backend=db_backend,
        sqlite_path=_optional("SQLITE_PATH", "/data/dflow.db") or "",
        postgres_host=_optional("POSTGRES_HOST"),
        postgres_port=_int("POSTGRES_PORT", 5432, minimum=1, maximum=65535),
        postgres_db=_optional("POSTGRES_DB"),
        postgres_user=_optional("POSTGRES_USER"),
        postgres_password=_optional("POSTGRES_PASSWORD"),
    )

    if db_backend == "postgres":
        for name, value in (
            ("POSTGRES_HOST", settings.postgres_host),
            ("POSTGRES_DB", settings.postgres_db),
            ("POSTGRES_USER", settings.postgres_user),
            ("POSTGRES_PASSWORD", settings.postgres_password),
        ):
            if not value:
                raise ConfigError(f"Missing required environment variable for DB_BACKEND=postgres: {name}")

    return settings
