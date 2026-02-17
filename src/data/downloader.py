"""Download and cache the AG News dataset from Hugging Face.

Provides utilities to fetch the full dataset and create smaller sample
subsets for development and CI testing.
"""

import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

LABEL_MAP: dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def download_ag_news(cache_dir: str = "data/raw") -> DatasetDict:
    """Download the AG News dataset from Hugging Face.

    Args:
        cache_dir: Directory for caching downloaded data.

    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    logger.info("Downloading AG News dataset to %s", cache_dir)
    dataset = load_dataset("ag_news", cache_dir=cache_dir)
    logger.info(
        "Dataset loaded: %d train, %d test",
        len(dataset["train"]),
        len(dataset["test"]),
    )
    return dataset


def create_sample_data(
    dataset: DatasetDict,
    n_samples: int = 200,
    output_dir: str = "data/sample",
) -> Dataset:
    """Create a small sample dataset for testing and CI.

    Args:
        dataset: Full dataset to sample from.
        n_samples: Number of samples to extract.
        output_dir: Directory to save the sample JSON.

    Returns:
        The sampled Dataset.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    samples = dataset["train"].shuffle(seed=42).select(range(n_samples))
    samples.to_json(f"{output_dir}/sample.json")
    logger.info("Created sample dataset with %d samples in %s", n_samples, output_dir)
    return samples
