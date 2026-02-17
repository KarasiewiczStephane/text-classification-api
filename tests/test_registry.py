"""Tests for the model registry with versioning."""

import json

import pytest

from src.models.registry import ModelRegistry, ModelVersion


@pytest.fixture
def registry_dir(tmp_path: str) -> str:
    """Create a temporary registry directory."""
    return str(tmp_path / "registry")


@pytest.fixture
def model_dir(tmp_path: str) -> str:
    """Create a fake model directory with metadata."""
    model_path = tmp_path / "fake_model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    metadata = {"num_parameters": 66_000_000}
    (model_path / "metadata.json").write_text(json.dumps(metadata))
    return str(model_path)


@pytest.fixture
def registry(registry_dir: str) -> ModelRegistry:
    """Create a fresh ModelRegistry instance."""
    return ModelRegistry(registry_dir=registry_dir)


class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_defaults(self) -> None:
        mv = ModelVersion(
            version="test_v1",
            model_type="distilbert",
            accuracy=0.92,
            training_time=100.0,
            num_parameters=66_000_000,
            created_at="2024-01-01",
            path="/models/test_v1",
        )
        assert mv.is_active is False
        assert mv.metadata == {}


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_creates_directory(self, registry_dir: str) -> None:
        from pathlib import Path

        ModelRegistry(registry_dir=registry_dir)
        assert Path(registry_dir).exists()

    def test_register_model(self, registry: ModelRegistry, model_dir: str) -> None:
        version = registry.register_model(
            model_path=model_dir,
            model_type="distilbert",
            accuracy=0.92,
            training_time=100.0,
        )
        assert version == "distilbert_v1"
        assert len(registry.list_models()) == 1

    def test_register_increments_version(self, registry: ModelRegistry, model_dir: str) -> None:
        v1 = registry.register_model(model_dir, "distilbert", 0.90, 50.0)
        v2 = registry.register_model(model_dir, "distilbert", 0.92, 60.0)
        assert v1 == "distilbert_v1"
        assert v2 == "distilbert_v2"

    def test_register_different_types(self, registry: ModelRegistry, model_dir: str) -> None:
        registry.register_model(model_dir, "distilbert", 0.90, 50.0)
        registry.register_model(model_dir, "bert", 0.92, 100.0)
        models = registry.list_models()
        assert len(models) == 2
        types = {m.model_type for m in models}
        assert types == {"distilbert", "bert"}

    def test_get_model(self, registry: ModelRegistry, model_dir: str) -> None:
        registry.register_model(model_dir, "distilbert", 0.92, 100.0)
        model = registry.get_model("distilbert_v1")
        assert model is not None
        assert model.accuracy == 0.92

    def test_get_model_not_found(self, registry: ModelRegistry) -> None:
        assert registry.get_model("nonexistent_v1") is None

    def test_list_models_empty(self, registry: ModelRegistry) -> None:
        assert registry.list_models() == []

    def test_set_active(self, registry: ModelRegistry, model_dir: str) -> None:
        registry.register_model(model_dir, "distilbert", 0.92, 100.0)
        result = registry.set_active("distilbert_v1")
        assert result is True
        active = registry.get_active_model()
        assert active is not None
        assert active.version == "distilbert_v1"

    def test_set_active_switches(self, registry: ModelRegistry, model_dir: str) -> None:
        registry.register_model(model_dir, "distilbert", 0.90, 50.0)
        registry.register_model(model_dir, "bert", 0.92, 100.0)
        registry.set_active("distilbert_v1")
        registry.set_active("bert_v1")
        active = registry.get_active_model()
        assert active.version == "bert_v1"
        # Only one should be active
        actives = [m for m in registry.list_models() if m.is_active]
        assert len(actives) == 1

    def test_set_active_nonexistent(self, registry: ModelRegistry) -> None:
        assert registry.set_active("fake_v99") is False

    def test_get_active_model_none(self, registry: ModelRegistry) -> None:
        assert registry.get_active_model() is None

    def test_compare_models(self, registry: ModelRegistry, model_dir: str) -> None:
        registry.register_model(model_dir, "distilbert", 0.90, 50.0)
        registry.register_model(model_dir, "bert", 0.92, 100.0)
        comparison = registry.compare_models("distilbert_v1", "bert_v1")
        assert "accuracy_diff" in comparison
        assert comparison["accuracy_diff"] == pytest.approx(-0.02)
        assert "speed_ratio" in comparison
        assert "size_ratio" in comparison

    def test_compare_models_not_found(self, registry: ModelRegistry) -> None:
        assert registry.compare_models("a", "b") == {}

    def test_persistence(self, registry_dir: str, model_dir: str) -> None:
        reg1 = ModelRegistry(registry_dir=registry_dir)
        reg1.register_model(model_dir, "distilbert", 0.92, 100.0)
        reg1.set_active("distilbert_v1")

        reg2 = ModelRegistry(registry_dir=registry_dir)
        assert len(reg2.list_models()) == 1
        assert reg2.get_active_model().version == "distilbert_v1"

    def test_reads_num_parameters_from_metadata(
        self, registry: ModelRegistry, model_dir: str
    ) -> None:
        registry.register_model(model_dir, "distilbert", 0.92, 100.0)
        model = registry.get_model("distilbert_v1")
        assert model.num_parameters == 66_000_000
