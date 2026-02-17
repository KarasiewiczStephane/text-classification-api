"""Model management endpoints for listing and switching active models."""

import logging

from fastapi import APIRouter, HTTPException

from src.api.routes.predict import inference
from src.api.schemas import ModelInfo
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter()
registry = ModelRegistry()


@router.get("", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List all available model versions."""
    models = registry.list_models()
    return [
        ModelInfo(
            version=m.version,
            model_type=m.model_type,
            accuracy=m.accuracy,
            is_active=m.is_active,
            created_at=m.created_at,
        )
        for m in models
    ]


@router.post("/switch")
async def switch_model(version: str) -> dict:
    """Switch the active model to a specified version.

    Args:
        version: Target model version identifier.
    """
    model = registry.get_model(version)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {version} not found")

    inference.load_model(model.path, model.version)
    registry.set_active(version)
    logger.info("Switched active model to %s", version)

    return {
        "status": "success",
        "active_model": version,
        "model_type": model.model_type,
    }


@router.get("/active", response_model=ModelInfo)
async def get_active_model() -> ModelInfo:
    """Get the currently active model version."""
    model = registry.get_active_model()
    if not model:
        raise HTTPException(status_code=404, detail="No active model")

    return ModelInfo(
        version=model.version,
        model_type=model.model_type,
        accuracy=model.accuracy,
        is_active=model.is_active,
        created_at=model.created_at,
    )
