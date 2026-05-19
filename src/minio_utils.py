import io
import os
from minio import Minio
from dotenv import load_dotenv

# Load variables from the .env file into the system environment
load_dotenv()

# Read the variables securely from the environment
MINIO_URL = os.getenv("MINIO_URL")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
# Initialize MinIO client
minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=True
)

if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)

def upload_csv_to_minio(file_content: bytes, experiment_id: str, filename: str) -> str:
    """Uploads the file to MinIO and returns the storage path."""
    minio_path = f"{experiment_id}/{filename}"
    
    minio_client.put_object(
        BUCKET_NAME,
        minio_path,
        io.BytesIO(file_content),
        length=len(file_content)
    )
    
    return minio_path