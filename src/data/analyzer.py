"""Data analysis utilities for text classification datasets.

Provides class distribution, text length statistics, vocabulary analysis,
and full report generation for exploratory data analysis.
"""

import logging
from collections import Counter
from typing import Any

import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Analyzer for HuggingFace text classification datasets.

    Args:
        dataset: A HuggingFace Dataset with 'text' and 'label' columns.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def class_distribution(self) -> dict[int, int]:
        """Compute the number of samples per class label.

        Returns:
            Mapping of label index to sample count.
        """
        return dict(Counter(self.dataset["label"]))

    def text_length_stats(self) -> dict[str, float]:
        """Compute word-count statistics across all texts.

        Returns:
            Dictionary with min, max, mean, median, std, and percentile stats.
        """
        lengths = [len(text.split()) for text in self.dataset["text"]]
        if not lengths:
            return {k: 0.0 for k in ("min", "max", "mean", "median", "std", "p25", "p75", "p95")}
        return {
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "p25": float(np.percentile(lengths, 25)),
            "p75": float(np.percentile(lengths, 75)),
            "p95": float(np.percentile(lengths, 95)),
        }

    def vocabulary_analysis(self, top_n: int = 100) -> dict[str, Any]:
        """Analyze vocabulary size and frequency distribution.

        Args:
            top_n: Number of most frequent words to include.

        Returns:
            Dictionary with total words, unique words, and top word list.
        """
        all_words: list[str] = []
        for text in self.dataset["text"]:
            all_words.extend(text.lower().split())

        word_counts = Counter(all_words)
        return {
            "total_words": len(all_words),
            "unique_words": len(word_counts),
            "top_words": word_counts.most_common(top_n),
        }

    def generate_report(self) -> dict[str, Any]:
        """Generate a complete analysis report.

        Returns:
            Dictionary containing sample count, class distribution,
            text length stats, and vocabulary analysis.
        """
        logger.info("Generating analysis report for %d samples", len(self.dataset))
        return {
            "num_samples": len(self.dataset),
            "class_distribution": self.class_distribution(),
            "text_length_stats": self.text_length_stats(),
            "vocabulary": self.vocabulary_analysis(),
        }
