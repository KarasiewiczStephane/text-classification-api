"""Tests for the training script module."""

from unittest.mock import MagicMock, patch

from src.models.trainer import create_bert_trainer, create_distilbert_trainer


class TestTrainScript:
    """Tests for the training entry point and factory functions."""

    def test_distilbert_trainer_config(self) -> None:
        trainer = create_distilbert_trainer()
        assert trainer.config.model_name == "distilbert-base-uncased"
        assert trainer.config.batch_size == 32
        assert trainer.config.learning_rate == 2e-5
        assert trainer.config.epochs == 3

    def test_bert_trainer_config(self) -> None:
        trainer = create_bert_trainer()
        assert trainer.config.model_name == "bert-base-uncased"
        assert trainer.config.batch_size == 16
        assert trainer.config.learning_rate == 2e-5

    def test_bert_uses_smaller_batch_than_distilbert(self) -> None:
        bert = create_bert_trainer()
        distilbert = create_distilbert_trainer()
        assert bert.config.batch_size < distilbert.config.batch_size

    @patch("src.train.download_ag_news")
    @patch("src.train.create_distilbert_trainer")
    @patch("src.train.tokenize_dataset")
    @patch("src.train.create_stratified_splits")
    @patch("src.train.TextPreprocessor")
    def test_main_distilbert(
        self,
        mock_preprocessor_cls: MagicMock,
        mock_splits: MagicMock,
        mock_tokenize: MagicMock,
        mock_trainer_fn: MagicMock,
        mock_download: MagicMock,
    ) -> None:
        """Verify main() orchestrates the pipeline correctly for distilbert."""
        from datasets import Dataset

        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = lambda self, key: Dataset.from_dict(
            {"text": ["a"], "label": [0]}
        )
        mock_download.return_value = mock_dataset

        mock_pp = MagicMock()
        mock_preprocessor_cls.return_value = mock_pp
        mock_pp.preprocess_dataset.return_value = Dataset.from_dict({"text": ["a"], "label": [0]})

        mock_splits.return_value = {
            "train": Dataset.from_dict({"text": ["a"], "label": [0]}),
            "validation": Dataset.from_dict({"text": ["b"], "label": [1]}),
            "test": Dataset.from_dict({"text": ["c"], "label": [2]}),
        }

        mock_tokenizer = MagicMock()
        mock_tokenize.return_value = (mock_splits.return_value, mock_tokenizer)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "training_time": 10.0}
        mock_trainer_fn.return_value = mock_trainer

        from src.train import main

        result = main("distilbert")
        assert "train_loss" in result
        mock_trainer.train.assert_called_once()
        mock_trainer.save_model.assert_called_once()

    @patch("src.train.download_ag_news")
    @patch("src.train.create_bert_trainer")
    @patch("src.train.tokenize_dataset")
    @patch("src.train.create_stratified_splits")
    @patch("src.train.TextPreprocessor")
    def test_main_bert(
        self,
        mock_preprocessor_cls: MagicMock,
        mock_splits: MagicMock,
        mock_tokenize: MagicMock,
        mock_trainer_fn: MagicMock,
        mock_download: MagicMock,
    ) -> None:
        """Verify main() orchestrates the pipeline correctly for bert."""
        from datasets import Dataset

        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = lambda self, key: Dataset.from_dict(
            {"text": ["a"], "label": [0]}
        )
        mock_download.return_value = mock_dataset

        mock_pp = MagicMock()
        mock_preprocessor_cls.return_value = mock_pp
        mock_pp.preprocess_dataset.return_value = Dataset.from_dict({"text": ["a"], "label": [0]})

        mock_splits.return_value = {
            "train": Dataset.from_dict({"text": ["a"], "label": [0]}),
            "validation": Dataset.from_dict({"text": ["b"], "label": [1]}),
            "test": Dataset.from_dict({"text": ["c"], "label": [2]}),
        }

        mock_tokenizer = MagicMock()
        mock_tokenize.return_value = (mock_splits.return_value, mock_tokenizer)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.4, "training_time": 20.0}
        mock_trainer_fn.return_value = mock_trainer

        from src.train import main

        result = main("bert")
        assert "train_loss" in result
        mock_trainer_fn.assert_called_once()
