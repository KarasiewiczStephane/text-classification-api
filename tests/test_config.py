"""Tests for configuration management module."""

import pytest
import yaml

from src.utils.config import Config


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    """Reset the Config singleton before each test."""
    Config.reset()
    yield
    Config.reset()


@pytest.fixture
def config_file(tmp_path: str) -> str:
    """Create a temporary config YAML file."""
    config_data = {
        "project": {"name": "test-project", "version": "0.1.0"},
        "data": {"max_length": 128, "dataset_name": "ag_news"},
        "training": {"models": {"distilbert": {"learning_rate": 2e-5}}},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return str(config_path)


class TestConfigLoad:
    """Tests for Config.load()."""

    def test_load_valid_config(self, config_file: str) -> None:
        """Config loads successfully from a valid YAML file."""
        config = Config.load(config_file)
        assert config is not None

    def test_load_returns_singleton(self, config_file: str) -> None:
        """Subsequent calls return the same instance."""
        config1 = Config.load(config_file)
        config2 = Config.load(config_file)
        assert config1 is config2

    def test_load_invalid_path_raises(self) -> None:
        """Loading from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.load("/nonexistent/config.yaml")

    def test_reset_allows_reload(self, config_file: str) -> None:
        """After reset, a new instance is created."""
        config1 = Config.load(config_file)
        Config.reset()
        config2 = Config.load(config_file)
        assert config1 is not config2


class TestConfigGet:
    """Tests for Config.get()."""

    def test_get_top_level_key(self, config_file: str) -> None:
        """Retrieve a top-level dictionary."""
        config = Config.load(config_file)
        project = config.get("project")
        assert project["name"] == "test-project"

    def test_get_nested_key(self, config_file: str) -> None:
        """Retrieve a nested value with dot notation."""
        config = Config.load(config_file)
        assert config.get("data.max_length") == 128

    def test_get_deeply_nested_key(self, config_file: str) -> None:
        """Retrieve a deeply nested value."""
        config = Config.load(config_file)
        assert config.get("training.models.distilbert.learning_rate") == 2e-5

    def test_get_missing_key_returns_default(self, config_file: str) -> None:
        """Missing keys return the specified default."""
        config = Config.load(config_file)
        assert config.get("nonexistent.key", default=42) == 42

    def test_get_missing_key_returns_none(self, config_file: str) -> None:
        """Missing keys return None when no default is specified."""
        config = Config.load(config_file)
        assert config.get("nonexistent") is None

    def test_get_dataset_name(self, config_file: str) -> None:
        """Retrieve the dataset name from config."""
        config = Config.load(config_file)
        assert config.get("data.dataset_name") == "ag_news"
