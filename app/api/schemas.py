"""API request and response schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check response."""
    ready: bool
    device: str
    loaded_checkpoints: str
    effnet_loaded: bool
    swin_loaded: bool = False
    version: str


class PredictionRequest(BaseModel):
    """Prediction request parameters."""
    mode: Literal["fusion", "cnn", "effnet", "swin", "swin_patchcam"] = "fusion"
    token_stage: str = Field("7", description="Swin token stage: 7/14/28/56/hr")
    enhance: bool = Field(False, description="Enhanced contrast with percentile normalization")
    per_pixel: bool = Field(False, description="Use per-pixel alpha blending")
    alpha_min: float = Field(0.0, ge=0.0, le=1.0)
    alpha_max: float = Field(0.6, ge=0.0, le=1.0)
    
    @field_validator("alpha_max")
    @classmethod
    def validate_alpha_range(cls, v: float, info) -> float:
        alpha_min = info.data.get("alpha_min", 0.0)
        if v < alpha_min:
            raise ValueError("alpha_max must be >= alpha_min")
        return v


class PredictionResponse(BaseModel):
    """Prediction response."""
    pred_label: str
    pred_idx: int = Field(description="Predicted class index")
    confidence: float = Field(ge=0.0, le=1.0, description="Prediction confidence")
    probs: dict[str, float]
    uploaded_url: str
    gradcam_url: str
    mode: str
    processing_time_ms: float | None = None


class MultiModelPredictionResponse(BaseModel):
    """Response for multi-model prediction."""
    fusion: PredictionResponse | dict
    effnet: PredictionResponse | dict
    swin: PredictionResponse | dict
    uploaded_url: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str | None = None
    status_code: int
    timestamp: str
