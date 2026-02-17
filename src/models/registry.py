"""Model registry for versioned model management.

Tracks multiple model versions with metadata, supports version switching,
and provides model comparison utilities.
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a registered model version.

    Args:
        version: Version identifier string.
        model_type: Model architecture type ('distilbert' or 'bert').
        accuracy: Evaluation accuracy on test set.
        training_time: Wall-clock training time in seconds.
        num_parameters: Total trainable parameter count.
        created_at: ISO-format creation timestamp.
        path: Filesystem path to the model directory.
        is_active: Whether this version is currently serving.
        metadata: Additional key-value metadata.
    """

    version: str
    model_type: str
    accuracy: float
    training_time: float
    num_parameters: int
    created_at: str
    path: str
    is_active: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Registry for managing multiple versioned models.

    Args:
        registry_dir: Directory for storing registered model copies.
    """

    def __init__(self, registry_dir: str = "models/registry") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self.versions: dict[str, ModelVersion] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load the registry index from disk."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data = json.load(f)
                self.versions = {k: ModelVersion(**v) for k, v in data.items()}
            logger.info("Loaded registry with %d models", len(self.versions))
        else:
            self.versions = {}

    def _save_registry(self) -> None:
        """Persist the registry index to disk."""
        with open(self.registry_file, "w") as f:
            json.dump({k: asdict(v) for k, v in self.versions.items()}, f, indent=2)

    def register_model(
        self,
        model_path: str,
        model_type: str,
        accuracy: float,
        training_time: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a new model version in the registry.

        Args:
            model_path: Path to the trained model directory.
            model_type: Architecture type ('distilbert' or 'bert').
            accuracy: Model accuracy on evaluation set.
            training_time: Training wall-clock time in seconds.
            metadata: Optional extra metadata.

        Returns:
            The assigned version string.
        """
        existing = [v for v in self.versions if v.startswith(model_type)]
        version_num = len(existing) + 1
        version = f"{model_type}_v{version_num}"

        dest_path = self.registry_dir / version
        shutil.copytree(model_path, dest_path, dirs_exist_ok=True)

        model_metadata_path = Path(model_path) / "metadata.json"
        num_params = 0
        if model_metadata_path.exists():
            with open(model_metadata_path) as f:
                model_meta = json.load(f)
                num_params = model_meta.get("num_parameters", 0)

        self.versions[version] = ModelVersion(
            version=version,
            model_type=model_type,
            accuracy=accuracy,
            training_time=training_time,
            num_parameters=num_params,
            created_at=datetime.now().isoformat(),
            path=str(dest_path),
            metadata=metadata or {},
        )

        self._save_registry()
        logger.info("Registered model %s (accuracy=%.4f)", version, accuracy)
        return version

    def list_models(self) -> list[ModelVersion]:
        """Return all registered model versions.

        Returns:
            List of ModelVersion instances.
        """
        return list(self.versions.values())

    def get_model(self, version: str) -> ModelVersion | None:
        """Retrieve a specific model version.

        Args:
            version: Version identifier string.

        Returns:
            ModelVersion if found, None otherwise.
        """
        return self.versions.get(version)

    def get_active_model(self) -> ModelVersion | None:
        """Get the currently active model version.

        Returns:
            The active ModelVersion, or None if no model is active.
        """
        for v in self.versions.values():
            if v.is_active:
                return v
        return None

    def set_active(self, version: str) -> bool:
        """Set a specific version as the active serving model.

        Args:
            version: Version identifier to activate.

        Returns:
            True if successful, False if the version was not found.
        """
        if version not in self.versions:
            return False

        for v in self.versions.values():
            v.is_active = False
        self.versions[version].is_active = True
        self._save_registry()
        logger.info("Set active model to %s", version)
        return True

    def compare_models(self, version_a: str, version_b: str) -> dict[str, float]:
        """Compare two model versions on key metrics.

        Args:
            version_a: First model version.
            version_b: Second model version.

        Returns:
            Dictionary with accuracy difference, speed ratio, and size ratio.
            Empty dict if either version is not found.
        """
        a = self.versions.get(version_a)
        b = self.versions.get(version_b)
        if not a or not b:
            return {}

        return {
            "accuracy_diff": a.accuracy - b.accuracy,
            "speed_ratio": b.training_time / a.training_time if a.training_time else 0,
            "size_ratio": b.num_parameters / a.num_parameters if a.num_parameters else 0,
        }
