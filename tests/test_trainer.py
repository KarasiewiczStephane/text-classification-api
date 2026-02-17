"""Tests for the model training pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.models.trainer import (
    ModelTrainer,
    TrainingConfig,
    create_bert_trainer,
    create_distilbert_trainer,
    get_device,
)


@pytest.fixture
def training_config() -> TrainingConfig:
    """Create a minimal training config for testing."""
    return TrainingConfig(
        model_name="distilbert-base-uncased",
        output_dir="/tmp/test_training",
        learning_rate=2e-5,
        batch_size=8,
        epochs=1,
    )


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self) -> None:
        config = TrainingConfig(model_name="test", output_dir="/tmp")
        assert config.learning_rate == 2e-5
        assert config.batch_size == 32
        assert config.epochs == 3
        assert config.warmup_ratio == 0.1
        assert config.early_stopping_patience == 2

    def test_custom_values(self) -> None:
        config = TrainingConfig(
            model_name="bert",
            output_dir="/out",
            learning_rate=1e-4,
            batch_size=16,
            epochs=5,
        )
        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
        assert config.epochs == 5


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def test_init(self, training_config: TrainingConfig) -> None:
        trainer = ModelTrainer(training_config)
        assert trainer.config is training_config
        assert trainer.num_labels == 4
        assert trainer.model is None
        assert trainer.training_time == 0.0

    def test_init_custom_labels(self, training_config: TrainingConfig) -> None:
        trainer = ModelTrainer(training_config, num_labels=10)
        assert trainer.num_labels == 10

    @patch("src.models.trainer.AutoModelForSequenceClassification")
    def test_load_model(self, mock_model_cls: MagicMock, training_config: TrainingConfig) -> None:
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        trainer = ModelTrainer(training_config)
        result = trainer.load_model()

        mock_model_cls.from_pretrained.assert_called_once_with(
            "distilbert-base-uncased", num_labels=4
        )
        assert result is mock_model
        assert trainer.model is mock_model

    @patch("src.models.trainer.AutoTokenizer")
    @patch("src.models.trainer.AutoModelForSequenceClassification")
    def test_save_model(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
        training_config: TrainingConfig,
        tmp_path: str,
    ) -> None:
        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        trainer = ModelTrainer(training_config)
        trainer.load_model()
        trainer.training_time = 10.5

        save_path = str(tmp_path / "saved_model")
        trainer.save_model(save_path, {"accuracy": 0.92})

        mock_model.save_pretrained.assert_called_once_with(save_path)

        with open(f"{save_path}/metadata.json") as f:
            metadata = json.load(f)
        assert metadata["accuracy"] == 0.92
        assert metadata["training_time"] == 10.5
        assert metadata["model_name"] == "distilbert-base-uncased"


class TestFactoryFunctions:
    """Tests for create_distilbert_trainer and create_bert_trainer."""

    def test_create_distilbert_trainer(self) -> None:
        trainer = create_distilbert_trainer()
        assert trainer.config.model_name == "distilbert-base-uncased"
        assert trainer.config.batch_size == 32
        assert trainer.num_labels == 4

    def test_create_bert_trainer(self) -> None:
        trainer = create_bert_trainer()
        assert trainer.config.model_name == "bert-base-uncased"
        assert trainer.config.batch_size == 16
        assert trainer.num_labels == 4

    def test_custom_output_dir(self) -> None:
        trainer = create_distilbert_trainer(output_dir="/custom/path")
        assert trainer.config.output_dir == "/custom/path"


class TestGetDevice:
    """Tests for get_device()."""

    def test_returns_device(self) -> None:
        device = get_device()
        assert str(device) in ("cpu", "cuda")
