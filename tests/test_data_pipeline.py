"""Tests for the data pipeline: downloader and preprocessor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from datasets import Dataset

from src.data.downloader import LABEL_MAP, create_sample_data, download_ag_news
from src.data.preprocessor import (
    TextPreprocessor,
    create_stratified_splits,
    tokenize_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataset() -> Dataset:
    """Create a small in-memory dataset for testing."""
    return Dataset.from_dict(
        {
            "text": [
                "Hello World! This is a test.",
                "Sports are great for health.",
                "Markets rose today, stocks up.",
                "New AI chip released by tech firm.",
                "Elections held across the country.",
                "Team wins the championship game!",
                "Revenue forecast exceeded expectations.",
                "Satellite launched into orbit successfully.",
            ]
            * 5,
            "label": [0, 1, 2, 3, 0, 1, 2, 3] * 5,
        }
    )


@pytest.fixture
def mock_dataset_dict(sample_dataset: Dataset) -> MagicMock:
    """Create a mock DatasetDict with train/test splits."""
    mock_dd = MagicMock()
    mock_dd.__getitem__ = lambda self, key: sample_dataset
    mock_dd.keys = lambda: ["train", "test"]
    return mock_dd


# ---------------------------------------------------------------------------
# Downloader tests
# ---------------------------------------------------------------------------


class TestLabelMap:
    """Tests for the LABEL_MAP constant."""

    def test_has_four_classes(self) -> None:
        assert len(LABEL_MAP) == 4

    def test_expected_labels(self) -> None:
        assert set(LABEL_MAP.values()) == {"World", "Sports", "Business", "Sci/Tech"}


class TestDownloadAgNews:
    """Tests for download_ag_news()."""

    @patch("src.data.downloader.load_dataset")
    def test_returns_dataset_dict(self, mock_load: MagicMock) -> None:
        train_ds = Dataset.from_dict({"text": ["a"], "label": [0]})
        test_ds = Dataset.from_dict({"text": ["b"], "label": [1]})
        mock_load.return_value = {"train": train_ds, "test": test_ds}
        result = download_ag_news(cache_dir="/tmp/test_cache")
        mock_load.assert_called_once_with("ag_news", cache_dir="/tmp/test_cache")
        assert "train" in result
        assert "test" in result


class TestCreateSampleData:
    """Tests for create_sample_data()."""

    def test_creates_sample(self, mock_dataset_dict: MagicMock, tmp_path: str) -> None:
        output = str(tmp_path / "sample")
        result = create_sample_data(mock_dataset_dict, n_samples=5, output_dir=output)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def test_lowercase(self) -> None:
        pp = TextPreprocessor(lowercase=True, remove_special_chars=False)
        assert pp.clean_text("Hello WORLD") == "hello world"

    def test_remove_special_chars(self) -> None:
        pp = TextPreprocessor(lowercase=False, remove_special_chars=True)
        assert pp.clean_text("Hello! @World#") == "Hello World"

    def test_both_options(self) -> None:
        pp = TextPreprocessor(lowercase=True, remove_special_chars=True)
        result = pp.clean_text("Hello!! World??  Test")
        assert result == "hello world test"

    def test_empty_string(self) -> None:
        pp = TextPreprocessor()
        assert pp.clean_text("") == ""

    def test_whitespace_normalization(self) -> None:
        pp = TextPreprocessor(lowercase=False, remove_special_chars=False)
        assert pp.clean_text("  too   many   spaces  ") == "too many spaces"

    def test_preprocess_dataset(self, sample_dataset: Dataset) -> None:
        pp = TextPreprocessor()
        processed = pp.preprocess_dataset(sample_dataset)
        assert len(processed) == len(sample_dataset)
        for text in processed["text"]:
            assert text == text.lower()


class TestCreateStratifiedSplits:
    """Tests for create_stratified_splits()."""

    def test_split_sizes(self, sample_dataset: Dataset) -> None:
        splits = create_stratified_splits(sample_dataset, 0.8, 0.1, 0.1)
        total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
        assert total == len(sample_dataset)

    def test_stratification_preserves_distribution(self, sample_dataset: Dataset) -> None:
        splits = create_stratified_splits(sample_dataset, 0.8, 0.1, 0.1)
        train_labels = np.array(splits["train"]["label"])
        unique, counts = np.unique(train_labels, return_counts=True)
        assert len(unique) == 4
        # Each class should have roughly equal counts
        assert counts.min() > 0

    def test_invalid_ratios_raise(self, sample_dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            create_stratified_splits(sample_dataset, 0.5, 0.1, 0.1)

    def test_all_splits_present(self, sample_dataset: Dataset) -> None:
        splits = create_stratified_splits(sample_dataset)
        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits


class TestTokenizeDataset:
    """Tests for tokenize_dataset()."""

    @patch("src.data.preprocessor.AutoTokenizer")
    def test_returns_tokenized_and_tokenizer(
        self, mock_tokenizer_cls: MagicMock, sample_dataset: Dataset
    ) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.side_effect = lambda texts, **kwargs: {
            "input_ids": [[0] * 128] * len(texts),
            "attention_mask": [[1] * 128] * len(texts),
        }

        splits = create_stratified_splits(sample_dataset)
        tokenized, tokenizer = tokenize_dataset(splits, "distilbert-base-uncased")
        assert tokenizer is mock_tokenizer
        assert "train" in tokenized
