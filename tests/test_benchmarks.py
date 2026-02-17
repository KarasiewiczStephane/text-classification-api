"""Tests for the benchmarking suite."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmarks.runner import (
    BenchmarkResults,
    BenchmarkRunner,
    generate_benchmark_report,
)


@pytest.fixture
def sample_results() -> list[BenchmarkResults]:
    """Create sample benchmark results for report generation."""
    return [
        BenchmarkResults(
            model_name="DistilBERT",
            latency_p50=12.0,
            latency_p95=20.0,
            latency_p99=25.0,
            throughput={1: 80.0, 8: 200.0, 16: 300.0, 32: 400.0},
            memory_mb=265.0,
            accuracy=0.921,
        ),
        BenchmarkResults(
            model_name="BERT",
            latency_p50=24.0,
            latency_p95=40.0,
            latency_p99=48.0,
            throughput={1: 40.0, 8: 100.0, 16: 150.0, 32: 200.0},
            memory_mb=420.0,
            accuracy=0.928,
        ),
    ]


class TestBenchmarkResults:
    """Tests for BenchmarkResults dataclass."""

    def test_create_results(self) -> None:
        r = BenchmarkResults(
            model_name="test",
            latency_p50=10.0,
            latency_p95=15.0,
            latency_p99=20.0,
            throughput={1: 100.0},
            memory_mb=100.0,
            accuracy=0.9,
        )
        assert r.model_name == "test"
        assert r.latency_p50 == 10.0


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_init(self) -> None:
        runner = BenchmarkRunner("/fake/path", "TestModel")
        assert runner.model_path == "/fake/path"
        assert runner.model_name == "TestModel"
        assert runner.model is None

    @patch("src.benchmarks.runner.AutoTokenizer")
    @patch("src.benchmarks.runner.AutoModelForSequenceClassification")
    def test_load_model(self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        runner = BenchmarkRunner("/fake/path", "TestModel")
        runner.load_model()

        assert runner.model is mock_model
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()

    @patch("src.benchmarks.runner.AutoTokenizer")
    @patch("src.benchmarks.runner.AutoModelForSequenceClassification")
    def test_measure_latency_returns_percentiles(
        self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        enc_output = MagicMock()
        enc_output.to = MagicMock(return_value=enc_output)
        mock_tokenizer.return_value = enc_output
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        runner = BenchmarkRunner("/fake/path", "TestModel")
        runner.load_model()

        texts = ["test text"] * 20
        latency = runner.measure_latency(texts, n_runs=20)

        assert "p50" in latency
        assert "p95" in latency
        assert "p99" in latency
        assert "mean" in latency
        assert "std" in latency
        assert latency["p50"] >= 0
        assert latency["p95"] >= latency["p50"]

    @patch("src.benchmarks.runner.AutoTokenizer")
    @patch("src.benchmarks.runner.AutoModelForSequenceClassification")
    def test_measure_throughput(
        self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        enc_output = MagicMock()
        enc_output.to = MagicMock(return_value=enc_output)
        mock_tokenizer.return_value = enc_output
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        runner = BenchmarkRunner("/fake/path", "TestModel")
        runner.load_model()

        texts = ["test text"] * 100
        throughput = runner.measure_throughput(texts, batch_sizes=[1, 8])

        assert 1 in throughput
        assert 8 in throughput
        assert throughput[1] > 0

    @patch("src.benchmarks.runner.psutil.Process")
    @patch("src.benchmarks.runner.AutoTokenizer")
    @patch("src.benchmarks.runner.AutoModelForSequenceClassification")
    def test_measure_memory(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
        mock_process: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        mock_mem = MagicMock()
        mock_mem.rss = 500 * 1024 * 1024  # 500 MB
        mock_process.return_value.memory_info.return_value = mock_mem

        runner = BenchmarkRunner("/fake/path", "TestModel")
        runner.load_model()
        memory = runner.measure_memory()
        assert memory > 0


class TestGenerateBenchmarkReport:
    """Tests for generate_benchmark_report."""

    def test_creates_report_and_charts(
        self, sample_results: list[BenchmarkResults], tmp_path: str
    ) -> None:
        output_dir = str(tmp_path / "benchmarks")
        report = generate_benchmark_report(sample_results, output_dir=output_dir)

        assert "# Benchmark Report" in report
        assert "DistilBERT" in report
        assert "BERT" in report
        assert Path(f"{output_dir}/benchmark_report.md").exists()
        assert Path(f"{output_dir}/latency_comparison.png").exists()
        assert Path(f"{output_dir}/throughput_comparison.png").exists()

    def test_report_contains_metrics(
        self, sample_results: list[BenchmarkResults], tmp_path: str
    ) -> None:
        output_dir = str(tmp_path / "benchmarks")
        report = generate_benchmark_report(sample_results, output_dir=output_dir)

        assert "92.10%" in report
        assert "92.80%" in report
        assert "12.00" in report
        assert "265.0" in report
