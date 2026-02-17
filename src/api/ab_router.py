"""A/B testing router with configurable traffic splitting and metrics.

Manages traffic distribution between two model versions and tracks
per-model latency, confidence, and label distribution metrics.
"""

import logging
import random
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ABMetrics:
    """Per-model A/B test metrics accumulator.

    Attributes:
        total_requests: Number of requests routed to this model.
        total_latency: Cumulative latency in milliseconds.
        confidence_sum: Cumulative prediction confidence.
        predictions: Per-label prediction counts.
    """

    total_requests: int = 0
    total_latency: float = 0.0
    confidence_sum: float = 0.0
    predictions: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def avg_latency(self) -> float:
        """Average request latency in milliseconds."""
        return self.total_latency / self.total_requests if self.total_requests else 0

    @property
    def avg_confidence(self) -> float:
        """Average prediction confidence."""
        return self.confidence_sum / self.total_requests if self.total_requests else 0


class ABRouter:
    """Traffic-splitting router for A/B testing between two models.

    Thread-safe router that distributes requests based on a configurable
    split ratio and accumulates per-model metrics.
    """

    def __init__(self) -> None:
        self.model_a: str | None = None
        self.model_b: str | None = None
        self.split_ratio: float = 0.8
        self.metrics: dict[str, ABMetrics] = {}
        self.enabled: bool = False
        self._lock = threading.Lock()

    def configure(self, model_a: str, model_b: str, split_ratio: float = 0.8) -> None:
        """Configure an A/B test between two models.

        Args:
            model_a: Primary model version identifier.
            model_b: Secondary model version identifier.
            split_ratio: Fraction of traffic routed to model_a (0.0-1.0).
        """
        with self._lock:
            self.model_a = model_a
            self.model_b = model_b
            self.split_ratio = max(0.0, min(1.0, split_ratio))
            self.metrics[model_a] = ABMetrics()
            self.metrics[model_b] = ABMetrics()
            self.enabled = True
        logger.info(
            "A/B test configured: %s (%.0f%%) vs %s (%.0f%%)",
            model_a,
            split_ratio * 100,
            model_b,
            (1 - split_ratio) * 100,
        )

    def get_model(self) -> str:
        """Select a model based on the traffic split ratio.

        Returns:
            Version identifier for the selected model.
        """
        if not self.enabled:
            return self.model_a
        return self.model_a if random.random() < self.split_ratio else self.model_b

    def record_result(self, model: str, latency: float, confidence: float, label: str) -> None:
        """Record a prediction result for metrics tracking.

        Args:
            model: Model version that served the request.
            latency: Request latency in milliseconds.
            confidence: Prediction confidence score.
            label: Predicted class label.
        """
        with self._lock:
            if model not in self.metrics:
                self.metrics[model] = ABMetrics()
            m = self.metrics[model]
            m.total_requests += 1
            m.total_latency += latency
            m.confidence_sum += confidence
            m.predictions[label] += 1

    def get_config(self) -> dict[str, Any]:
        """Get the current A/B test configuration.

        Returns:
            Dictionary with enabled status, model names, and split ratio.
        """
        return {
            "enabled": self.enabled,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "split_ratio": self.split_ratio,
        }

    def get_results(self) -> dict[str, dict[str, Any]]:
        """Get per-model A/B test results.

        Returns:
            Dictionary mapping model version to aggregated metrics.
        """
        results = {}
        for model, metrics in self.metrics.items():
            results[model] = {
                "total_requests": metrics.total_requests,
                "avg_latency_ms": metrics.avg_latency,
                "avg_confidence": metrics.avg_confidence,
                "label_distribution": dict(metrics.predictions),
            }
        return results

    def reset_metrics(self) -> None:
        """Reset all accumulated metrics."""
        with self._lock:
            for model in self.metrics:
                self.metrics[model] = ABMetrics()
        logger.info("A/B test metrics reset")


# Global router instance
ab_router = ABRouter()
