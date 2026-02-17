"""Benchmark runner for model latency, throughput, and memory profiling.

Measures p50/p95/p99 latency, throughput at various batch sizes,
memory footprint, and generates comparison reports with charts.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results for a single model.

    Attributes:
        model_name: Human-readable model identifier.
        latency_p50: 50th percentile latency in milliseconds.
        latency_p95: 95th percentile latency in milliseconds.
        latency_p99: 99th percentile latency in milliseconds.
        throughput: Mapping of batch_size to texts/second.
        memory_mb: Peak memory usage in megabytes.
        accuracy: Model accuracy for comparison.
    """

    model_name: str
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: dict[int, float]
    memory_mb: float
    accuracy: float


class BenchmarkRunner:
    """Runner for comprehensive model performance benchmarks.

    Args:
        model_path: Path to the saved model directory.
        model_name: Human-readable name for reporting.
    """

    def __init__(self, model_path: str, model_name: str) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: AutoModelForSequenceClassification | None = None
        self.tokenizer: AutoTokenizer | None = None

    def load_model(self) -> None:
        """Load the model and tokenizer from disk."""
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded %s for benchmarking on %s", self.model_name, self.device)

    def measure_latency(self, texts: list[str], n_runs: int = 100) -> dict[str, float]:
        """Measure single-text inference latency percentiles.

        Args:
            texts: Pool of texts to sample from.
            n_runs: Number of inference runs to measure.

        Returns:
            Dictionary with p50, p95, p99, mean, and std in milliseconds.
        """
        latencies: list[float] = []

        for _ in range(10):
            self._infer_single(texts[0])

        for text in texts[:n_runs]:
            start = time.perf_counter()
            self._infer_single(text)
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
        }

    def _infer_single(self, text: str) -> None:
        """Run inference on a single text."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            self.model(**inputs)

    def measure_throughput(
        self,
        texts: list[str],
        batch_sizes: list[int] | None = None,
    ) -> dict[int, float]:
        """Measure inference throughput at different batch sizes.

        Args:
            texts: Pool of texts to batch from.
            batch_sizes: List of batch sizes to test.

        Returns:
            Mapping of batch_size to texts processed per second.
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32]

        results: dict[int, float] = {}
        for batch_size in batch_sizes:
            batches = [
                texts[i : i + batch_size] for i in range(0, min(len(texts), 320), batch_size)
            ]

            start = time.perf_counter()
            for batch in batches:
                self._infer_batch(batch)
            elapsed = time.perf_counter() - start

            total_texts = len(batches) * batch_size
            results[batch_size] = total_texts / elapsed if elapsed > 0 else 0
        return results

    def _infer_batch(self, texts: list[str]) -> None:
        """Run inference on a batch of texts."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            self.model(**inputs)

    def measure_memory(self) -> float:
        """Measure model memory footprint in megabytes.

        Returns:
            Memory usage in MB (GPU peak or RSS for CPU).
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self._infer_single("test text for memory measurement")
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024

    def run_full_benchmark(self, texts: list[str], accuracy: float) -> BenchmarkResults:
        """Execute all benchmarks and return aggregated results.

        Args:
            texts: Pool of texts for benchmarking.
            accuracy: Model accuracy for the comparison table.

        Returns:
            BenchmarkResults with all metrics.
        """
        if self.model is None:
            self.load_model()

        latency = self.measure_latency(texts)
        throughput = self.measure_throughput(texts)
        memory = self.measure_memory()

        logger.info(
            "%s benchmark: p50=%.1fms, p95=%.1fms, memory=%.0fMB",
            self.model_name,
            latency["p50"],
            latency["p95"],
            memory,
        )

        return BenchmarkResults(
            model_name=self.model_name,
            latency_p50=latency["p50"],
            latency_p95=latency["p95"],
            latency_p99=latency["p99"],
            throughput=throughput,
            memory_mb=memory,
            accuracy=accuracy,
        )


def generate_benchmark_report(
    results: list[BenchmarkResults],
    output_dir: str = "docs/benchmarks",
) -> str:
    """Generate a Markdown report with comparison tables and charts.

    Args:
        results: List of BenchmarkResults from different models.
        output_dir: Directory to save the report and chart images.

    Returns:
        The generated Markdown report string.
    """
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Latency comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [r.model_name for r in results]
    p50 = [r.latency_p50 for r in results]
    p95 = [r.latency_p95 for r in results]
    p99 = [r.latency_p99 for r in results]

    x = np.arange(len(models))
    width = 0.25
    ax.bar(x - width, p50, width, label="p50")
    ax.bar(x, p95, width, label="p95")
    ax.bar(x + width, p99, width, label="p99")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_title("Inference Latency Comparison")
    plt.savefig(f"{output_dir}/latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Throughput chart
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        batch_sizes = list(r.throughput.keys())
        throughputs = list(r.throughput.values())
        ax.plot(batch_sizes, throughputs, marker="o", label=r.model_name)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (texts/sec)")
    ax.legend()
    ax.set_title("Throughput vs Batch Size")
    plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Markdown report
    report = "# Benchmark Report\n\n## Model Comparison\n\n"
    report += "| Model | Accuracy | p50 (ms) | p95 (ms) | p99 (ms) | Memory (MB) |\n"
    report += "|-------|----------|----------|----------|----------|-------------|\n"
    for r in results:
        report += (
            f"| {r.model_name} | {r.accuracy:.2%} | {r.latency_p50:.2f} "
            f"| {r.latency_p95:.2f} | {r.latency_p99:.2f} | {r.memory_mb:.1f} |\n"
        )

    report += "\n## Latency Comparison\n\n![Latency](latency_comparison.png)\n"
    report += "\n## Throughput Comparison\n\n![Throughput](throughput_comparison.png)\n"

    with open(f"{output_dir}/benchmark_report.md", "w") as f:
        f.write(report)

    logger.info("Benchmark report saved to %s", output_dir)
    return report
