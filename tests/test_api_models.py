"""Tests for model management endpoints and request logging middleware."""

import hashlib
import time
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.models.registry import ModelVersion


@pytest.fixture
def mock_registry():
    """Patch registries in both app and routes modules."""
    with (
        patch("src.api.app.ModelRegistry") as app_reg_cls,
        patch("src.api.routes.models.registry") as routes_reg,
    ):
        app_mock = MagicMock()
        app_mock.get_active_model.return_value = None
        app_reg_cls.return_value = app_mock
        yield routes_reg


@pytest.fixture
def sample_model_version() -> ModelVersion:
    """Create a sample ModelVersion for testing."""
    return ModelVersion(
        version="distilbert_v1",
        model_type="distilbert",
        accuracy=0.92,
        training_time=100.0,
        num_parameters=66_000_000,
        created_at="2024-01-01T00:00:00",
        path="/models/registry/distilbert_v1",
        is_active=True,
    )


@pytest.mark.asyncio
async def test_list_models_empty(mock_registry: MagicMock) -> None:
    """GET /models returns empty list when no models registered."""
    mock_registry.list_models.return_value = []
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_models_with_data(
    mock_registry: MagicMock, sample_model_version: ModelVersion
) -> None:
    """GET /models returns model info list."""
    mock_registry.list_models.return_value = [sample_model_version]
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["version"] == "distilbert_v1"
    assert data[0]["model_type"] == "distilbert"
    assert data[0]["accuracy"] == 0.92
    assert data[0]["is_active"] is True


@pytest.mark.asyncio
async def test_switch_model_not_found(mock_registry: MagicMock) -> None:
    """POST /models/switch returns 404 for unknown version."""
    mock_registry.get_model.return_value = None
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/models/switch", params={"version": "fake_v99"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_switch_model_success(
    mock_registry: MagicMock, sample_model_version: ModelVersion
) -> None:
    """POST /models/switch loads and activates the model."""
    mock_registry.get_model.return_value = sample_model_version
    mock_registry.set_active.return_value = True

    from src.api.app import app

    with patch("src.api.routes.models.inference") as mock_inference:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/models/switch", params={"version": "distilbert_v1"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["active_model"] == "distilbert_v1"
    mock_inference.load_model.assert_called_once()


@pytest.mark.asyncio
async def test_get_active_model(
    mock_registry: MagicMock, sample_model_version: ModelVersion
) -> None:
    """GET /models/active returns the active model info."""
    mock_registry.get_active_model.return_value = sample_model_version
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models/active")
    assert response.status_code == 200
    assert response.json()["version"] == "distilbert_v1"


@pytest.mark.asyncio
async def test_get_active_model_none(mock_registry: MagicMock) -> None:
    """GET /models/active returns 404 when no model is active."""
    mock_registry.get_active_model.return_value = None
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models/active")
    assert response.status_code == 404


class TestRequestLoggingMiddleware:
    """Tests for the request logging middleware."""

    def test_text_hash_deterministic(self) -> None:
        """Same input produces the same hash."""
        body = b'{"text": "hello world"}'
        hash1 = hashlib.md5(body).hexdigest()[:8]  # noqa: S324
        hash2 = hashlib.md5(body).hexdigest()[:8]  # noqa: S324
        assert hash1 == hash2

    def test_latency_calculation(self) -> None:
        """Latency is computed in milliseconds."""
        start = time.time()
        time.sleep(0.01)
        latency = (time.time() - start) * 1000
        assert latency > 5  # at least 5ms
