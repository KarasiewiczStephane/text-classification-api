"""Tests for the model evaluation suite."""

from unittest.mock import MagicMock, patch

from src.models.evaluator import LABEL_NAMES, ModelEvaluator, plot_confusion_matrix


class TestLabelNames:
    """Tests for the LABEL_NAMES constant."""

    def test_has_four_labels(self) -> None:
        assert len(LABEL_NAMES) == 4

    def test_expected_values(self) -> None:
        assert LABEL_NAMES == ["World", "Sports", "Business", "Sci/Tech"]


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    @patch("src.models.evaluator.AutoTokenizer")
    @patch("src.models.evaluator.AutoModelForSequenceClassification")
    def test_init_loads_model(
        self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        evaluator = ModelEvaluator("/fake/path", device="cpu")
        assert evaluator.device == "cpu"
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()

    @patch("src.models.evaluator.AutoTokenizer")
    @patch("src.models.evaluator.AutoModelForSequenceClassification")
    def test_predict_returns_correct_shapes(
        self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock
    ) -> None:
        import torch

        mock_model = MagicMock()
        logits = torch.tensor([[0.1, 0.8, 0.05, 0.05], [0.7, 0.1, 0.1, 0.1]])
        mock_model.return_value = MagicMock(logits=logits)
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        enc_output = MagicMock()
        enc_output.to = MagicMock(return_value=enc_output)
        mock_tokenizer.return_value = enc_output
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        evaluator = ModelEvaluator("/fake/path", device="cpu")
        preds, probs = evaluator.predict(["text1", "text2"])

        assert preds.shape == (2,)
        assert probs.shape == (2, 4)
        assert preds[0] == 1  # highest prob at index 1
        assert preds[1] == 0  # highest prob at index 0

    @patch("src.models.evaluator.AutoTokenizer")
    @patch("src.models.evaluator.AutoModelForSequenceClassification")
    def test_evaluate_returns_all_metrics(
        self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock
    ) -> None:
        import torch

        # 8 samples with known logits to produce deterministic predictions
        logits_batch = torch.tensor(
            [
                [0.9, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.0],
                [0.0, 0.0, 0.0, 0.9],
                [0.9, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.0],
                [0.0, 0.0, 0.0, 0.9],
            ]
        )

        mock_model = MagicMock()
        mock_model.return_value = MagicMock(logits=logits_batch)
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        dummy_enc = MagicMock()
        dummy_enc.to = MagicMock(return_value=dummy_enc)
        mock_tokenizer.return_value = dummy_enc
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        evaluator = ModelEvaluator("/fake/path", device="cpu")

        dataset = MagicMock()
        dataset.__getitem__ = lambda self, key: {
            "text": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "label": [0, 1, 2, 3, 0, 1, 2, 3],
        }[key]

        results = evaluator.evaluate(dataset)
        assert "accuracy" in results
        assert results["accuracy"] == 1.0
        assert "per_class_metrics" in results
        assert len(results["per_class_metrics"]) == 4
        assert "confusion_matrix" in results
        assert len(results["confusion_matrix"]) == 4
        assert "classification_report" in results

    @patch("src.models.evaluator.AutoTokenizer")
    @patch("src.models.evaluator.AutoModelForSequenceClassification")
    def test_save_report(
        self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock, tmp_path: str
    ) -> None:
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        evaluator = ModelEvaluator("/fake/path", device="cpu")
        report_path = str(tmp_path / "reports" / "eval.json")
        evaluator.save_report({"accuracy": 0.95}, report_path)

        import json

        with open(report_path) as f:
            data = json.load(f)
        assert data["accuracy"] == 0.95


class TestPlotConfusionMatrix:
    """Tests for the plot_confusion_matrix function."""

    def test_creates_png_file(self, tmp_path: str) -> None:
        cm = [[10, 1, 0, 0], [0, 12, 1, 0], [1, 0, 11, 0], [0, 0, 1, 9]]
        output_path = str(tmp_path / "cm.png")
        plot_confusion_matrix(cm, LABEL_NAMES, output_path)
        from pathlib import Path

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0
