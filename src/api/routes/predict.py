"""Prediction endpoints for single and batch text classification."""

import asyncio
import logging

import torch
from fastapi import APIRouter, HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

LABEL_NAMES: list[str] = ["World", "Sports", "Business", "Sci/Tech"]


class ModelInference:
    """Handles model loading and inference for the API.

    Manages a single loaded model and provides sync/async predict methods.
    """

    def __init__(self) -> None:
        self.model: AutoModelForSequenceClassification | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.model_version: str | None = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold: float = 0.5

    def load_model(self, model_path: str, version: str) -> None:
        """Load a model from disk for serving.

        Args:
            model_path: Path to the saved model directory.
            version: Version identifier string.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.model_version = version
        logger.info("Loaded model %s from %s", version, model_path)

    async def predict(self, text: str) -> PredictResponse:
        """Async single-text prediction.

        Args:
            text: Input text to classify.

        Returns:
            PredictResponse with label, confidence, and probabilities.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_sync, text)

    def _predict_sync(self, text: str) -> PredictResponse:
        """Synchronous single-text prediction."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        probs_np = probs.cpu().numpy()
        pred_idx = int(probs_np.argmax())
        confidence = float(probs_np[pred_idx])

        return PredictResponse(
            label=LABEL_NAMES[pred_idx],
            confidence=confidence,
            probabilities={LABEL_NAMES[i]: float(probs_np[i]) for i in range(4)},
            model_version=self.model_version,
            uncertain=confidence < self.confidence_threshold,
        )

    async def predict_batch(self, texts: list[str]) -> list[PredictResponse]:
        """Async batch prediction.

        Args:
            texts: List of input texts to classify.

        Returns:
            List of PredictResponse objects.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_batch_sync, texts)

    def _predict_batch_sync(self, texts: list[str]) -> list[PredictResponse]:
        """Synchronous batch prediction."""
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

        results = []
        for prob in probs.cpu().numpy():
            pred_idx = int(prob.argmax())
            confidence = float(prob[pred_idx])
            results.append(
                PredictResponse(
                    label=LABEL_NAMES[pred_idx],
                    confidence=confidence,
                    probabilities={LABEL_NAMES[j]: float(prob[j]) for j in range(4)},
                    model_version=self.model_version,
                    uncertain=confidence < self.confidence_threshold,
                )
            )
        return results


# Global inference instance
inference = ModelInference()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Classify a single text input."""
    if inference.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return await inference.predict(request.text)


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Classify a batch of text inputs."""
    if inference.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    predictions = await inference.predict_batch(request.texts)
    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions),
        model_version=inference.model_version,
    )
