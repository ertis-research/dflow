import io
from typing import Optional

from minio import Minio
from minio.error import MinioException
from urllib3.exceptions import HTTPError as Urllib3HTTPError

from src.config import get_settings

_client: Optional[Minio] = None

# What the MinIO client raises on failure: S3Error/ServerError/InvalidResponseError (all
# MinioException) for error responses, and urllib3 errors when the server is unreachable.
MINIO_ERRORS = (MinioException, Urllib3HTTPError)


def get_minio_client() -> Minio:
    global _client
    if _client is None:
        settings = get_settings()
        _client = Minio(
            settings.minio_url,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
    return _client


def ensure_bucket() -> None:
    """Creates the dataset bucket if it does not exist. Called once at startup."""
    bucket = get_settings().bucket_name
    client = get_minio_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def training_data_prefix(training_id: str) -> str:
    return f"{training_id}/"


def upload_csv_to_minio(file_content: bytes, training_id: str, filename: str) -> str:
    """Uploads the file to MinIO and returns the storage path."""
    minio_path = f"{training_data_prefix(training_id)}{filename}"

    get_minio_client().put_object(
        get_settings().bucket_name,
        minio_path,
        io.BytesIO(file_content),
        length=len(file_content)
    )

    return minio_path


def delete_training_data(training_id: str) -> int:
    """Deletes every object uploaded for a training. Returns how many objects were removed."""
    bucket = get_settings().bucket_name
    client = get_minio_client()
    removed = 0
    for obj in client.list_objects(bucket, prefix=training_data_prefix(training_id), recursive=True):
        if obj.object_name is None:
            continue
        client.remove_object(bucket, obj.object_name)
        removed += 1
    return removed
