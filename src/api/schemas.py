"""Pydantic request/response schemas for the API endpoints."""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Single text classification request.

    Attributes:
        text: Input text to classify (1-512 characters).
    """

    text: str = Field(..., min_length=1, max_length=512)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Reject whitespace-only input."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class PredictResponse(BaseModel):
    """Classification result for a single text.

    Attributes:
        label: Predicted class name.
        confidence: Probability of the predicted class.
        probabilities: Per-class probability distribution.
        model_version: Model version that produced the prediction.
        uncertain: True if confidence is below the threshold.
    """

    label: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    uncertain: bool = False


class BatchPredictRequest(BaseModel):
    """Batch text classification request.

    Attributes:
        texts: List of input texts (1-50 items).
    """

    texts: list[str] = Field(..., min_length=1, max_length=50)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        """Strip and filter empty strings."""
        validated = [t.strip() for t in v if t.strip()]
        if not validated:
            raise ValueError("At least one non-empty text required")
        if len(validated) > 50:
            raise ValueError("Maximum 50 texts per batch")
        return validated


class BatchPredictResponse(BaseModel):
    """Batch classification results.

    Attributes:
        predictions: List of individual prediction results.
        total: Number of predictions returned.
        model_version: Model version used.
    """

    predictions: list[PredictResponse]
    total: int
    model_version: str


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Service health status.
        model_version: Currently loaded model version.
        model_loaded: Whether a model is loaded and ready.
    """

    status: str
    model_version: str
    model_loaded: bool


class ModelInfo(BaseModel):
    """Summary information about a registered model.

    Attributes:
        version: Model version identifier.
        model_type: Architecture type.
        accuracy: Evaluation accuracy.
        is_active: Whether this model is currently active.
        created_at: Registration timestamp.
    """

    version: str
    model_type: str
    accuracy: float
    is_active: bool
    created_at: str
