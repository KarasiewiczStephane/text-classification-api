"""Tests for the text classification dashboard data generators."""

import numpy as np
import pandas as pd

from src.dashboard.app import (
    CATEGORIES,
    generate_ab_metrics,
    generate_confusion_data,
    generate_latency_trend,
    generate_prediction_distribution,
)


class TestAbMetrics:
    def test_returns_dataframe(self) -> None:
        df = generate_ab_metrics()
        assert isinstance(df, pd.DataFrame)

    def test_has_two_models(self) -> None:
        df = generate_ab_metrics()
        assert len(df) == 2

    def test_has_required_columns(self) -> None:
        df = generate_ab_metrics()
        for col in ["model", "requests", "avg_latency_ms", "accuracy", "avg_confidence"]:
            assert col in df.columns

    def test_accuracy_bounded(self) -> None:
        df = generate_ab_metrics()
        assert (df["accuracy"] >= 0).all()
        assert (df["accuracy"] <= 1).all()

    def test_latencies_positive(self) -> None:
        df = generate_ab_metrics()
        assert (df["avg_latency_ms"] > 0).all()

    def test_reproducible(self) -> None:
        df1 = generate_ab_metrics(seed=99)
        df2 = generate_ab_metrics(seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestPredictionDistribution:
    def test_returns_dataframe(self) -> None:
        df = generate_prediction_distribution()
        assert isinstance(df, pd.DataFrame)

    def test_has_all_categories(self) -> None:
        df = generate_prediction_distribution()
        assert len(df) == len(CATEGORIES)

    def test_counts_positive(self) -> None:
        df = generate_prediction_distribution()
        assert (df["count"] > 0).all()

    def test_confidence_bounded(self) -> None:
        df = generate_prediction_distribution()
        assert (df["avg_confidence"] >= 0).all()
        assert (df["avg_confidence"] <= 1).all()


class TestLatencyTrend:
    def test_returns_dataframe(self) -> None:
        df = generate_latency_trend()
        assert isinstance(df, pd.DataFrame)

    def test_has_entries(self) -> None:
        df = generate_latency_trend()
        assert len(df) == 96  # 48 timestamps x 2 models

    def test_latency_positive(self) -> None:
        df = generate_latency_trend()
        assert (df["latency_ms"] > 0).all()

    def test_two_models(self) -> None:
        df = generate_latency_trend()
        assert df["model"].nunique() == 2


class TestConfusionData:
    def test_returns_ndarray(self) -> None:
        cm = generate_confusion_data()
        assert isinstance(cm, np.ndarray)

    def test_correct_shape(self) -> None:
        cm = generate_confusion_data()
        n = len(CATEGORIES)
        assert cm.shape == (n, n)

    def test_non_negative(self) -> None:
        cm = generate_confusion_data()
        assert (cm >= 0).all()

    def test_diagonal_dominant(self) -> None:
        cm = generate_confusion_data()
        for i in range(cm.shape[0]):
            assert cm[i, i] >= cm[i].sum() * 0.5
