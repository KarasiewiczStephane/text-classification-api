"""Model training pipeline with early stopping and warmup scheduling.

Supports fine-tuning HuggingFace transformer models for sequence
classification with configurable hyperparameters and model saving.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for a model training run.

    Args:
        model_name: HuggingFace model identifier.
        output_dir: Directory for training checkpoints.
        learning_rate: Optimizer learning rate.
        batch_size: Training and eval batch size.
        epochs: Maximum number of training epochs.
        warmup_ratio: Fraction of steps for linear warmup.
        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_threshold: Minimum improvement to count as progress.
    """

    model_name: str
    output_dir: str
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 3
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.001


class ModelTrainer:
    """Trainer for fine-tuning HuggingFace classification models.

    Args:
        config: Training configuration.
        num_labels: Number of output classes.
    """

    def __init__(self, config: TrainingConfig, num_labels: int = 4) -> None:
        self.config = config
        self.num_labels = num_labels
        self.model: AutoModelForSequenceClassification | None = None
        self.training_time: float = 0.0

    def load_model(self) -> AutoModelForSequenceClassification:
        """Load the pre-trained model for sequence classification.

        Returns:
            The loaded model with the classification head.
        """
        logger.info("Loading model: %s with %d labels", self.config.model_name, self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.num_labels,
        )
        return self.model

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Any,
        tokenizer: AutoTokenizer,
    ) -> dict[str, Any]:
        """Run the training loop with early stopping.

        Args:
            train_dataset: Tokenized training dataset.
            eval_dataset: Tokenized evaluation dataset.
            tokenizer: Tokenizer matching the model.

        Returns:
            Dictionary with training loss, time, and steps completed.
        """
        if self.model is None:
            self.load_model()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=100,
            report_to="none",
        )

        def compute_metrics(eval_pred: Any) -> dict[str, float]:
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=-1)
            accuracy = (predictions == labels).mean()
            return {"accuracy": float(accuracy)}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
            ],
        )

        start_time = time.time()
        train_result = trainer.train()
        self.training_time = time.time() - start_time

        logger.info(
            "Training complete in %.1fs, loss=%.4f",
            self.training_time,
            train_result.training_loss,
        )

        return {
            "train_loss": train_result.training_loss,
            "training_time": self.training_time,
            "epochs_completed": train_result.global_step,
        }

    def save_model(self, path: str, metadata: dict[str, Any]) -> None:
        """Save the trained model and metadata to disk.

        Args:
            path: Directory to save the model files.
            metadata: Additional metadata to persist alongside the model.
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.save_pretrained(path)

        metadata.update(
            {
                "training_time": self.training_time,
                "model_name": self.config.model_name,
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
            }
        )

        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model saved to %s", path)


def create_distilbert_trainer(output_dir: str = "models/distilbert") -> ModelTrainer:
    """Create a trainer configured for DistilBERT.

    Args:
        output_dir: Output directory for checkpoints.

    Returns:
        Configured ModelTrainer instance.
    """
    config = TrainingConfig(
        model_name="distilbert-base-uncased",
        output_dir=output_dir,
        learning_rate=2e-5,
        batch_size=32,
        epochs=3,
        warmup_ratio=0.1,
        early_stopping_patience=2,
    )
    return ModelTrainer(config, num_labels=4)


def create_bert_trainer(output_dir: str = "models/bert") -> ModelTrainer:
    """Create a trainer configured for BERT with smaller batch size.

    Args:
        output_dir: Output directory for checkpoints.

    Returns:
        Configured ModelTrainer instance.
    """
    config = TrainingConfig(
        model_name="bert-base-uncased",
        output_dir=output_dir,
        learning_rate=2e-5,
        batch_size=16,
        epochs=3,
        warmup_ratio=0.1,
        early_stopping_patience=2,
    )
    return ModelTrainer(config, num_labels=4)


def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Returns:
        torch.device for CUDA if available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device
