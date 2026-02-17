"""A/B testing configuration and results endpoints (placeholder for Task 10)."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/config")
async def get_ab_config() -> dict:
    """Get A/B test configuration (placeholder)."""
    return {"enabled": False, "model_a": None, "model_b": None, "split_ratio": 0.8}
