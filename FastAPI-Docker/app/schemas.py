"""Response schemas exposed by the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ChestXRayClass = Literal["COVID", "NORMAL", "PNEUMONIA"]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]


class PredictionResponse(BaseModel):
    predicted_class: ChestXRayClass
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[ChestXRayClass, float]


class GradCAMResponse(PredictionResponse):
    gradcam_image_url: str = Field(
        description="Temporary URL that serves the generated Grad-CAM PNG image.",
    )
