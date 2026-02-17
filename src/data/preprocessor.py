"""Text preprocessing and tokenization for classification datasets.

Handles text cleaning, stratified train/val/test splitting, and
HuggingFace tokenizer integration for transformer models.
"""

import logging
import re

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Configurable text cleaner for classification pipelines.

    Args:
        lowercase: Whether to convert text to lowercase.
        remove_special_chars: Whether to strip non-alphanumeric characters.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars

    def clean_text(self, text: str) -> str:
        """Clean a single text string.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text with normalized whitespace.
        """
        if self.lowercase:
            text = text.lower()
        if self.remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Apply text cleaning to an entire dataset.

        Args:
            dataset: HuggingFace Dataset with a 'text' column.

        Returns:
            Dataset with cleaned text column.
        """
        logger.info("Preprocessing %d samples", len(dataset))
        return dataset.map(lambda x: {"text": self.clean_text(x["text"])})


def create_stratified_splits(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Create stratified train/validation/test splits.

    Args:
        dataset: Source dataset with a 'label' column.
        train_ratio: Fraction for the training set.
        val_ratio: Fraction for the validation set.
        test_ratio: Fraction for the test set.
        seed: Random seed for reproducibility.

    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits.

    Raises:
        ValueError: If ratios do not sum to 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    labels = np.array(dataset["label"])
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, stratify=labels, random_state=seed
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, stratify=labels[temp_idx], random_state=seed
    )

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    return DatasetDict(
        {
            "train": dataset.select(train_idx.tolist()),
            "validation": dataset.select(val_idx.tolist()),
            "test": dataset.select(test_idx.tolist()),
        }
    )


def tokenize_dataset(
    dataset: DatasetDict,
    model_name: str,
    max_length: int = 128,
) -> tuple[DatasetDict, AutoTokenizer]:
    """Tokenize a dataset for transformer model training.

    Args:
        dataset: DatasetDict with text splits.
        model_name: HuggingFace model identifier for the tokenizer.
        max_length: Maximum sequence length for padding/truncation.

    Returns:
        Tuple of (tokenized DatasetDict, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    logger.info("Tokenized dataset with model %s, max_length=%d", model_name, max_length)
    return tokenized, tokenizer
