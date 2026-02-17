"""Training script for fine-tuning DistilBERT or BERT on AG News.

Usage:
    python -m src.train --model distilbert
    python -m src.train --model bert
"""

import argparse

from src.data.downloader import download_ag_news
from src.data.preprocessor import (
    TextPreprocessor,
    create_stratified_splits,
    tokenize_dataset,
)
from src.models.trainer import create_bert_trainer, create_distilbert_trainer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main(model_type: str = "distilbert") -> dict:
    """Run the full training pipeline for the specified model.

    Args:
        model_type: One of 'distilbert' or 'bert'.

    Returns:
        Training results dictionary.
    """
    logger.info("Starting training pipeline for %s", model_type)

    dataset = download_ag_news()
    preprocessor = TextPreprocessor()
    processed = preprocessor.preprocess_dataset(dataset["train"])
    splits = create_stratified_splits(processed)

    if model_type == "bert":
        trainer = create_bert_trainer()
        model_name = "bert-base-uncased"
    else:
        trainer = create_distilbert_trainer()
        model_name = "distilbert-base-uncased"

    tokenized, tokenizer = tokenize_dataset(splits, model_name)

    results = trainer.train(
        tokenized["train"],
        tokenized["validation"],
        tokenizer,
    )

    trainer.save_model(f"models/{model_type}_v1", results)
    logger.info("Training complete: %s", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classification model")
    parser.add_argument(
        "--model",
        choices=["bert", "distilbert"],
        default="distilbert",
        help="Model architecture to fine-tune",
    )
    args = parser.parse_args()
    main(args.model)
