"""Tests for the data analysis module."""

import pytest
from datasets import Dataset

from src.data.analyzer import DataAnalyzer


@pytest.fixture
def analysis_dataset() -> Dataset:
    """Create a small dataset for analysis testing."""
    return Dataset.from_dict(
        {
            "text": [
                "hello world this is a test",
                "sports news about the game",
                "stock market update for today",
                "new technology released today",
                "global politics news update",
                "football match results today",
                "business revenue report quarterly",
                "science breakthrough in lab",
            ],
            "label": [0, 1, 2, 3, 0, 1, 2, 3],
        }
    )


@pytest.fixture
def analyzer(analysis_dataset: Dataset) -> DataAnalyzer:
    """Create a DataAnalyzer instance."""
    return DataAnalyzer(analysis_dataset)


class TestClassDistribution:
    """Tests for DataAnalyzer.class_distribution()."""

    def test_returns_all_classes(self, analyzer: DataAnalyzer) -> None:
        dist = analyzer.class_distribution()
        assert len(dist) == 4

    def test_correct_counts(self, analyzer: DataAnalyzer) -> None:
        dist = analyzer.class_distribution()
        assert all(count == 2 for count in dist.values())

    def test_single_class_dataset(self) -> None:
        ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 0]})
        analyzer = DataAnalyzer(ds)
        dist = analyzer.class_distribution()
        assert dist == {0: 2}


class TestTextLengthStats:
    """Tests for DataAnalyzer.text_length_stats()."""

    def test_returns_expected_keys(self, analyzer: DataAnalyzer) -> None:
        stats = analyzer.text_length_stats()
        expected_keys = {"min", "max", "mean", "median", "std", "p25", "p75", "p95"}
        assert set(stats.keys()) == expected_keys

    def test_min_less_than_max(self, analyzer: DataAnalyzer) -> None:
        stats = analyzer.text_length_stats()
        assert stats["min"] <= stats["max"]

    def test_mean_in_range(self, analyzer: DataAnalyzer) -> None:
        stats = analyzer.text_length_stats()
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_values_are_numeric(self, analyzer: DataAnalyzer) -> None:
        stats = analyzer.text_length_stats()
        for value in stats.values():
            assert isinstance(value, int | float)


class TestVocabularyAnalysis:
    """Tests for DataAnalyzer.vocabulary_analysis()."""

    def test_returns_expected_keys(self, analyzer: DataAnalyzer) -> None:
        vocab = analyzer.vocabulary_analysis()
        assert "total_words" in vocab
        assert "unique_words" in vocab
        assert "top_words" in vocab

    def test_total_words_positive(self, analyzer: DataAnalyzer) -> None:
        vocab = analyzer.vocabulary_analysis()
        assert vocab["total_words"] > 0

    def test_unique_less_or_equal_total(self, analyzer: DataAnalyzer) -> None:
        vocab = analyzer.vocabulary_analysis()
        assert vocab["unique_words"] <= vocab["total_words"]

    def test_top_n_limit(self, analyzer: DataAnalyzer) -> None:
        vocab = analyzer.vocabulary_analysis(top_n=3)
        assert len(vocab["top_words"]) <= 3

    def test_top_words_are_tuples(self, analyzer: DataAnalyzer) -> None:
        vocab = analyzer.vocabulary_analysis(top_n=5)
        for item in vocab["top_words"]:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestGenerateReport:
    """Tests for DataAnalyzer.generate_report()."""

    def test_report_contains_all_sections(self, analyzer: DataAnalyzer) -> None:
        report = analyzer.generate_report()
        assert "num_samples" in report
        assert "class_distribution" in report
        assert "text_length_stats" in report
        assert "vocabulary" in report

    def test_num_samples_correct(self, analyzer: DataAnalyzer) -> None:
        report = analyzer.generate_report()
        assert report["num_samples"] == 8

    def test_empty_dataset(self) -> None:
        ds = Dataset.from_dict({"text": [], "label": []})
        analyzer = DataAnalyzer(ds)
        report = analyzer.generate_report()
        assert report["num_samples"] == 0
