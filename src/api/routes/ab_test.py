"""A/B testing configuration and results endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.ab_router import ab_router
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter()
registry = ModelRegistry()


class ABConfigRequest(BaseModel):
    """Request body for configuring an A/B test.

    Attributes:
        model_a: Primary model version.
        model_b: Secondary model version.
        split_ratio: Fraction of traffic to model_a (0.0-1.0).
    """

    model_a: str
    model_b: str
    split_ratio: float = Field(0.8, ge=0.0, le=1.0)


class ABConfigResponse(BaseModel):
    """Response body for A/B test configuration.

    Attributes:
        enabled: Whether A/B testing is active.
        model_a: Primary model version.
        model_b: Secondary model version.
        split_ratio: Current traffic split ratio.
    """

    enabled: bool
    model_a: str | None
    model_b: str | None
    split_ratio: float


@router.get("/config", response_model=ABConfigResponse)
async def get_ab_config() -> ABConfigResponse:
    """Get the current A/B test configuration."""
    config = ab_router.get_config()
    return ABConfigResponse(**config)


@router.put("/config")
async def update_ab_config(request: ABConfigRequest) -> dict:
    """Configure a new A/B test between two models.

    Args:
        request: A/B test configuration with model versions and split ratio.
    """
    if not registry.get_model(request.model_a):
        raise HTTPException(404, f"Model {request.model_a} not found")
    if not registry.get_model(request.model_b):
        raise HTTPException(404, f"Model {request.model_b} not found")

    ab_router.configure(request.model_a, request.model_b, request.split_ratio)
    return {"status": "configured", **ab_router.get_config()}


@router.get("/results")
async def get_ab_results() -> dict:
    """Get per-model A/B test metrics."""
    return ab_router.get_results()


@router.post("/reset")
async def reset_ab_metrics() -> dict:
    """Reset all accumulated A/B test metrics."""
    ab_router.reset_metrics()
    return {"status": "metrics reset"}
