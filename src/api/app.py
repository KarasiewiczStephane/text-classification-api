"""FastAPI application factory and startup configuration."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import ab_test, models, predict
from src.api.routes.predict import inference
from src.api.schemas import HealthResponse
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the active model on startup, clean up on shutdown."""
    active = registry.get_active_model()
    if active:
        inference.load_model(active.path, active.version)
        logger.info("Loaded active model %s on startup", active.version)
    else:
        logger.warning("No active model found in registry")
    yield


app = FastAPI(
    title="Text Classification API",
    description="Multi-class text classification using DistilBERT/BERT",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health status and model info."""
    return HealthResponse(
        status="healthy",
        model_version=inference.model_version or "none",
        model_loaded=inference.model is not None,
    )


app.include_router(predict.router, tags=["Prediction"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(ab_test.router, prefix="/ab-test", tags=["A/B Testing"])
