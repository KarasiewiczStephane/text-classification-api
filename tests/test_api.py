"""Tests for the FastAPI application and prediction endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from src.api.schemas import (
    BatchPredictRequest,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestPredictRequest:
    """Tests for PredictRequest schema validation."""

    def test_valid_text(self) -> None:
        req = PredictRequest(text="Hello world")
        assert req.text == "Hello world"

    def test_strips_whitespace(self) -> None:
        req = PredictRequest(text="  padded text  ")
        assert req.text == "padded text"

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(text="")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(text="   ")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(text="a" * 513)

    def test_max_length_ok(self) -> None:
        req = PredictRequest(text="a" * 512)
        assert len(req.text) == 512


class TestBatchPredictRequest:
    """Tests for BatchPredictRequest schema validation."""

    def test_valid_batch(self) -> None:
        req = BatchPredictRequest(texts=["text1", "text2"])
        assert len(req.texts) == 2

    def test_strips_texts(self) -> None:
        req = BatchPredictRequest(texts=["  hello  ", "world  "])
        assert req.texts == ["hello", "world"]

    def test_filters_empty_strings(self) -> None:
        req = BatchPredictRequest(texts=["hello", "", "  ", "world"])
        assert req.texts == ["hello", "world"]

    def test_rejects_all_empty(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictRequest(texts=["", "  "])

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictRequest(texts=[])


class TestPredictResponse:
    """Tests for PredictResponse schema."""

    def test_creates_response(self) -> None:
        resp = PredictResponse(
            label="Sports",
            confidence=0.95,
            probabilities={"World": 0.01, "Sports": 0.95, "Business": 0.02, "Sci/Tech": 0.02},
            model_version="distilbert_v1",
        )
        assert resp.label == "Sports"
        assert resp.uncertain is False

    def test_uncertain_flag(self) -> None:
        resp = PredictResponse(
            label="World",
            confidence=0.3,
            probabilities={"World": 0.3, "Sports": 0.3, "Business": 0.2, "Sci/Tech": 0.2},
            model_version="v1",
            uncertain=True,
        )
        assert resp.uncertain is True


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_creates_health(self) -> None:
        resp = HealthResponse(status="healthy", model_version="v1", model_loaded=True)
        assert resp.status == "healthy"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_inference():
    """Patch the inference singleton with a mock."""
    with patch("src.api.routes.predict.inference") as mock:
        mock.model = MagicMock()
        mock.model_version = "test_v1"
        yield mock


@pytest.fixture
def mock_registry():
    """Patch the registry with a mock."""
    with patch("src.api.app.ModelRegistry") as mock_cls:
        mock_reg = MagicMock()
        mock_reg.get_active_model.return_value = None
        mock_cls.return_value = mock_reg
        yield mock_reg


@pytest.mark.asyncio
async def test_health_endpoint(mock_registry: MagicMock) -> None:
    """Health endpoint returns status information."""
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_predict_no_model(mock_registry: MagicMock) -> None:
    """Predict returns 503 when no model is loaded."""
    from src.api.app import app
    from src.api.routes.predict import inference

    inference.model = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json={"text": "test"})
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_predict_batch_no_model(mock_registry: MagicMock) -> None:
    """Batch predict returns 503 when no model is loaded."""
    from src.api.app import app
    from src.api.routes.predict import inference

    inference.model = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict/batch", json={"texts": ["test"]})
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_predict_invalid_input(mock_registry: MagicMock) -> None:
    """Predict rejects empty text input."""
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json={"text": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ab_test_config(mock_registry: MagicMock) -> None:
    """AB test config endpoint returns default config."""
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ab-test/config")
    assert response.status_code == 200
    assert "enabled" in response.json()
