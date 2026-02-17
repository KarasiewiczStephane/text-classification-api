"""Tests for A/B testing router and endpoints."""

from collections import Counter
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.ab_router import ABMetrics, ABRouter


class TestABMetrics:
    """Tests for ABMetrics dataclass."""

    def test_avg_latency_zero_requests(self) -> None:
        m = ABMetrics()
        assert m.avg_latency == 0

    def test_avg_latency(self) -> None:
        m = ABMetrics(total_requests=10, total_latency=100.0)
        assert m.avg_latency == 10.0

    def test_avg_confidence_zero_requests(self) -> None:
        m = ABMetrics()
        assert m.avg_confidence == 0

    def test_avg_confidence(self) -> None:
        m = ABMetrics(total_requests=5, confidence_sum=4.5)
        assert m.avg_confidence == pytest.approx(0.9)


class TestABRouter:
    """Tests for ABRouter class."""

    def test_init_disabled(self) -> None:
        router = ABRouter()
        assert router.enabled is False
        assert router.model_a is None

    def test_configure(self) -> None:
        router = ABRouter()
        router.configure("model_a", "model_b", 0.7)
        assert router.enabled is True
        assert router.model_a == "model_a"
        assert router.model_b == "model_b"
        assert router.split_ratio == 0.7

    def test_configure_clamps_ratio(self) -> None:
        router = ABRouter()
        router.configure("a", "b", 1.5)
        assert router.split_ratio == 1.0
        router.configure("a", "b", -0.5)
        assert router.split_ratio == 0.0

    def test_get_model_disabled(self) -> None:
        router = ABRouter()
        router.model_a = "only_model"
        assert router.get_model() == "only_model"

    def test_get_model_split(self) -> None:
        """Test traffic split roughly matches configured ratio."""
        router = ABRouter()
        router.configure("model_a", "model_b", 0.8)

        counts = Counter()
        n = 10000
        for _ in range(n):
            counts[router.get_model()] += 1

        ratio = counts["model_a"] / n
        assert 0.75 < ratio < 0.85  # within reasonable bounds

    def test_record_result(self) -> None:
        router = ABRouter()
        router.configure("a", "b")
        router.record_result("a", latency=10.0, confidence=0.9, label="Sports")
        router.record_result("a", latency=20.0, confidence=0.8, label="World")

        assert router.metrics["a"].total_requests == 2
        assert router.metrics["a"].total_latency == 30.0
        assert router.metrics["a"].predictions["Sports"] == 1
        assert router.metrics["a"].predictions["World"] == 1

    def test_get_config(self) -> None:
        router = ABRouter()
        router.configure("a", "b", 0.6)
        config = router.get_config()
        assert config["enabled"] is True
        assert config["split_ratio"] == 0.6

    def test_get_results(self) -> None:
        router = ABRouter()
        router.configure("a", "b")
        router.record_result("a", 10.0, 0.9, "Sports")
        results = router.get_results()
        assert "a" in results
        assert results["a"]["total_requests"] == 1
        assert results["a"]["avg_latency_ms"] == 10.0

    def test_reset_metrics(self) -> None:
        router = ABRouter()
        router.configure("a", "b")
        router.record_result("a", 10.0, 0.9, "Sports")
        router.reset_metrics()
        assert router.metrics["a"].total_requests == 0

    def test_record_unknown_model(self) -> None:
        router = ABRouter()
        router.record_result("unknown", 5.0, 0.7, "Business")
        assert router.metrics["unknown"].total_requests == 1


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ab_deps():
    """Patch registries and ab_router for endpoint tests."""
    with (
        patch("src.api.app.ModelRegistry") as app_reg_cls,
        patch("src.api.routes.ab_test.registry") as route_reg,
        patch("src.api.routes.ab_test.ab_router") as route_ab,
    ):
        app_mock = MagicMock()
        app_mock.get_active_model.return_value = None
        app_reg_cls.return_value = app_mock
        yield route_reg, route_ab


@pytest.mark.asyncio
async def test_ab_config_endpoint(mock_ab_deps: tuple) -> None:
    """GET /ab-test/config returns current config."""
    _, mock_ab = mock_ab_deps
    mock_ab.get_config.return_value = {
        "enabled": True,
        "model_a": "a",
        "model_b": "b",
        "split_ratio": 0.8,
    }
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ab-test/config")
    assert response.status_code == 200
    assert response.json()["enabled"] is True


@pytest.mark.asyncio
async def test_ab_results_endpoint(mock_ab_deps: tuple) -> None:
    """GET /ab-test/results returns per-model metrics."""
    _, mock_ab = mock_ab_deps
    mock_ab.get_results.return_value = {"a": {"total_requests": 5}}
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ab-test/results")
    assert response.status_code == 200
    assert "a" in response.json()


@pytest.mark.asyncio
async def test_ab_reset_endpoint(mock_ab_deps: tuple) -> None:
    """POST /ab-test/reset clears metrics."""
    _, mock_ab = mock_ab_deps
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ab-test/reset")
    assert response.status_code == 200
    mock_ab.reset_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_ab_configure_not_found(mock_ab_deps: tuple) -> None:
    """PUT /ab-test/config returns 404 for unknown model."""
    route_reg, _ = mock_ab_deps
    route_reg.get_model.return_value = None
    from src.api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.put(
            "/ab-test/config",
            json={"model_a": "fake_a", "model_b": "fake_b", "split_ratio": 0.5},
        )
    assert response.status_code == 404
