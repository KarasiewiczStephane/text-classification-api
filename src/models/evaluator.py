"""Model evaluation with per-class metrics, confusion matrix, and reporting.

Provides batch inference, comprehensive metric computation, and
JSON report generation for trained classification models.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

LABEL_NAMES: list[str] = ["World", "Sports", "Business", "Sci/Tech"]


class ModelEvaluator:
    """Evaluate a trained classification model on a dataset.

    Args:
        model_path: Path to the saved model directory.
        device: Compute device ('cuda' or 'cpu'). Auto-detected if None.
    """

    def __init__(self, model_path: str, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Loaded model from %s on %s", model_path, self.device)

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Get predictions and probabilities for a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            Tuple of (predicted class indices, probability arrays).
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)

        return preds.cpu().numpy(), probs.cpu().numpy()

    def evaluate(self, dataset: Any) -> dict[str, Any]:
        """Run comprehensive evaluation on a dataset.

        Args:
            dataset: HuggingFace Dataset with 'text' and 'label' columns.

        Returns:
            Dictionary with accuracy, per-class metrics, confusion matrix,
            and full classification report.
        """
        all_preds: list[int] = []
        all_labels: list[int] = []
        batch_size = 32
        texts = dataset["text"]
        labels = dataset["label"]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            preds, _ = self.predict(batch_texts)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels[i : i + batch_size])

        all_preds_arr = np.array(all_preds)
        all_labels_arr = np.array(all_labels)

        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels_arr, all_preds_arr, average=None
        )

        return {
            "accuracy": float(accuracy_score(all_labels_arr, all_preds_arr)),
            "per_class_metrics": {
                LABEL_NAMES[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(len(LABEL_NAMES))
            },
            "confusion_matrix": confusion_matrix(all_labels_arr, all_preds_arr).tolist(),
            "classification_report": classification_report(
                all_labels_arr, all_preds_arr, target_names=LABEL_NAMES, output_dict=True
            ),
        }

    def save_report(self, results: dict[str, Any], output_path: str) -> None:
        """Save evaluation results to a JSON file.

        Args:
            results: Evaluation results dictionary.
            output_path: File path for the JSON report.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation report saved to %s", output_path)


def plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str],
    output_path: str,
) -> None:
    """Generate and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix as a nested list.
        labels: Class label names for axes.
        output_path: File path to save the PNG chart.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Confusion matrix saved to %s", output_path)
