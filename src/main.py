import asyncio
import contextlib
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.experiments import router as experiments_router
from src.config import get_settings
from src.db.store import close_store, init_store
from src.integrations.minio_storage import ensure_bucket
from src.training.status import poll_training_status_forever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fails fast (and names the problem) if a required setting is missing,
    # the database is unreachable or MinIO cannot be used.
    settings = get_settings()
    if settings.mlflow_enabled:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    init_store(settings)
    ensure_bucket()

    poller = asyncio.create_task(poll_training_status_forever())
    try:
        yield
    finally:
        poller.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poller
        close_store()


app = FastAPI(
    title="dflow API",
    description="API to orchestrate ML model deployments and training",
    version="1.0.0",
    lifespan=lifespan,
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(experiments_router)
