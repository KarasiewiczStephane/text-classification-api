"""Integration tests for the API endpoints.

These tests run against a live API instance and are intended for CI
environments where the API is started as a background process.
"""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    """Import the FastAPI app for testing."""
    from src.api.app import app

    return app


@pytest.mark.asyncio
async def test_health_endpoint(app) -> None:
    """Health endpoint responds with status information."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


@pytest.mark.asyncio
async def test_openapi_docs(app) -> None:
    """OpenAPI docs are accessible."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/docs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_models_list(app) -> None:
    """Models endpoint returns a list."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
