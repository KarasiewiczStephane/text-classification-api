"""Configuration management using YAML files.

Provides a singleton Config class for loading and accessing nested
configuration values with dot-notation keys.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Singleton configuration manager that loads values from a YAML file.

    Usage:
        config = Config.load("configs/config.yaml")
        max_length = config.get("data.max_length", default=128)
    """

    _instance: Optional["Config"] = None
    _config: dict = {}

    def __init__(self) -> None:
        """Initialize empty Config instance."""

    @classmethod
    def load(cls, config_path: str = "configs/config.yaml") -> "Config":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            The singleton Config instance with loaded values.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        if cls._instance is None:
            cls._instance = cls()
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(path) as f:
                cls._instance._config = yaml.safe_load(f) or {}
            logger.info("Configuration loaded from %s", config_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance, allowing re-loading."""
        cls._instance = None
        cls._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value using dot-notation.

        Args:
            key: Dot-separated key path (e.g. 'data.max_length').
            default: Value to return if the key is not found.

        Returns:
            The configuration value, or the default if not found.
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(k, {})
        return value if value != {} else default
