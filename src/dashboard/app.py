"""Streamlit dashboard for text classification visualization.

Displays A/B test metrics, model comparison, prediction distribution,
and latency analysis using synthetic demo data.

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]


def generate_ab_metrics(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic A/B test metrics for two models."""
    rng = np.random.default_rng(seed)
    models = ["DistilBERT (Model A)", "BERT-base (Model B)"]
    rows = []
    for model in models:
        base_acc = 0.91 if "BERT-base" in model else 0.89
        rows.append(
            {
                "model": model,
                "requests": int(rng.integers(5000, 15000)),
                "avg_latency_ms": round(
                    rng.uniform(15, 35) if "DistilBERT" in model else rng.uniform(40, 80),
                    1,
                ),
                "p99_latency_ms": round(
                    rng.uniform(50, 80) if "DistilBERT" in model else rng.uniform(100, 180),
                    1,
                ),
                "accuracy": round(base_acc + rng.uniform(-0.02, 0.02), 4),
                "avg_confidence": round(rng.uniform(0.82, 0.95), 4),
            }
        )
    return pd.DataFrame(rows)


def generate_prediction_distribution(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic prediction distribution by category."""
    rng = np.random.default_rng(seed)
    rows = []
    for cat in CATEGORIES:
        count = int(rng.integers(1000, 5000))
        rows.append(
            {
                "category": cat,
                "count": count,
                "avg_confidence": round(rng.uniform(0.80, 0.96), 4),
            }
        )
    return pd.DataFrame(rows)


def generate_latency_trend(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic latency trend over time."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-11-28 00:00", periods=48, freq="30min")
    rows = []
    for ts in timestamps:
        for model in ["DistilBERT", "BERT-base"]:
            base = 25 if model == "DistilBERT" else 55
            rows.append(
                {
                    "timestamp": ts,
                    "model": model,
                    "latency_ms": round(base + rng.uniform(-8, 12), 1),
                    "requests": int(rng.integers(50, 300)),
                }
            )
    return pd.DataFrame(rows)


def generate_confusion_data(seed: int = 42) -> np.ndarray:
    """Generate synthetic confusion matrix."""
    rng = np.random.default_rng(seed)
    n = len(CATEGORIES)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        total = int(rng.integers(200, 500))
        correct = int(total * rng.uniform(0.85, 0.96))
        matrix[i, i] = correct
        remaining = total - correct
        for j in range(n):
            if j != i and remaining > 0:
                misclass = int(rng.integers(0, max(remaining // (n - 1) + 1, 1)))
                matrix[i, j] = misclass
                remaining -= misclass
    return matrix


def render_header() -> None:
    """Render the dashboard header."""
    st.title("Text Classification API Dashboard")
    st.caption(
        "Multi-class document classification with DistilBERT/BERT A/B testing, "
        "model versioning, and real-time performance monitoring"
    )


def render_summary_metrics(ab_df: pd.DataFrame, dist_df: pd.DataFrame) -> None:
    """Render top-level summary metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", f"{ab_df['requests'].sum():,}")
    best = ab_df.loc[ab_df["accuracy"].idxmax()]
    col2.metric("Best Accuracy", f"{best['accuracy']:.1%}")
    fastest = ab_df.loc[ab_df["avg_latency_ms"].idxmin()]
    col3.metric("Fastest Model", fastest["model"].split("(")[0].strip())
    col4.metric("Categories", len(dist_df))


def render_ab_comparison(ab_df: pd.DataFrame) -> None:
    """Render A/B test model comparison."""
    st.subheader("A/B Test: Model Comparison")
    fig = go.Figure()
    metrics = ["accuracy", "avg_confidence"]
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                name=metric.replace("_", " ").title(),
                x=ab_df["model"],
                y=ab_df[metric],
                text=ab_df[metric].apply(lambda x: f"{x:.3f}"),
                textposition="auto",
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0.7, 1.0]},
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_distribution(dist_df: pd.DataFrame) -> None:
    """Render prediction distribution pie chart."""
    st.subheader("Prediction Distribution")
    fig = px.pie(
        dist_df,
        values="count",
        names="category",
        color_discrete_sequence=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
    )
    fig.update_layout(
        height=350,
        margin={"l": 20, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_latency_trend(latency_df: pd.DataFrame) -> None:
    """Render latency trend over time."""
    st.subheader("Latency Trend (24h)")
    fig = px.line(
        latency_df,
        x="timestamp",
        y="latency_ms",
        color="model",
        markers=False,
    )
    fig.update_layout(
        yaxis_title="Latency (ms)",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(cm: np.ndarray) -> None:
    """Render confusion matrix heatmap."""
    st.subheader("Confusion Matrix")
    fig = px.imshow(
        cm,
        x=CATEGORIES,
        y=CATEGORIES,
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    ab_df = generate_ab_metrics()
    dist_df = generate_prediction_distribution()
    latency_df = generate_latency_trend()
    cm = generate_confusion_data()

    render_summary_metrics(ab_df, dist_df)
    st.markdown("---")

    render_ab_comparison(ab_df)

    col_left, col_right = st.columns(2)
    with col_left:
        render_prediction_distribution(dist_df)
    with col_right:
        render_latency_trend(latency_df)

    st.markdown("---")
    render_confusion_matrix(cm)


if __name__ == "__main__":
    main()
