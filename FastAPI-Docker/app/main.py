"""FastAPI application for DenseNet121 chest X-ray classification."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from threading import Lock
from time import monotonic

import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from tensorflow import keras

from .gradcam import GradCAMError, generate_gradcam
from .model import ModelLoadError, load_model
from .preprocessing import (
    MAX_UPLOAD_BYTES,
    ImageTooLargeError,
    InvalidImageError,
    preprocess_image,
)
from .schemas import GradCAMResponse, HealthResponse, PredictionResponse

logger = logging.getLogger(__name__)
CLASS_NAMES = ("COVID", "NORMAL", "PNEUMONIA")
GRADCAM_IMAGE_TTL_SECONDS = 10 * 60
MAX_STORED_GRADCAM_IMAGES = 256


class GradCAMImageStore:
    """Bounded, thread-safe, expiring in-memory storage for generated PNGs."""

    def __init__(self, ttl_seconds: int, max_items: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_items = max_items
        self._images: dict[str, tuple[float, bytes]] = {}
        self._lock = Lock()

    def _purge_expired(self, now: float) -> None:
        expired = [image_id for image_id, (expires_at, _) in self._images.items() if expires_at <= now]
        for image_id in expired:
            del self._images[image_id]

    def put(self, image_bytes: bytes) -> str:
        now = monotonic()
        with self._lock:
            self._purge_expired(now)
            while len(self._images) >= self._max_items:
                oldest_id = min(self._images, key=lambda image_id: self._images[image_id][0])
                del self._images[oldest_id]
            image_id = secrets.token_urlsafe(24)
            self._images[image_id] = (now + self._ttl_seconds, image_bytes)
            return image_id

    def get(self, image_id: str) -> bytes | None:
        now = monotonic()
        with self._lock:
            self._purge_expired(now)
            stored = self._images.get(image_id)
            return None if stored is None else stored[1]


gradcam_image_store = GradCAMImageStore(
    ttl_seconds=GRADCAM_IMAGE_TTL_SECONDS,
    max_items=MAX_STORED_GRADCAM_IMAGES,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once per application process."""
    app.state.model = None
    app.state.model_error = None
    try:
        app.state.model = await run_in_threadpool(load_model)
        logger.info("Chest X-ray model loaded successfully.")
    except ModelLoadError as exc:
        app.state.model_error = str(exc)
        logger.exception("Chest X-ray model could not be loaded.")
    yield
    app.state.model = None


app = FastAPI(
    title="Chest X-Ray Classification API",
    version="1.0.0",
    description="DenseNet121-based COVID, normal, and pneumonia X-ray classification.",
    lifespan=lifespan,
)


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    """Provide a small public API landing response."""
    return {
        "message": "Chest X-Ray Classification API is running",
        "docs": "/docs",
        "health": "/health",
    }


def get_loaded_model() -> keras.Model:
    if app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable. Check server logs for model loading details.",
        )
    return app.state.model


async def read_and_preprocess_upload(image: UploadFile | None) -> tuple[bytes, np.ndarray]:
    """Apply the existing upload limits and verified prediction preprocessing."""
    if image is None or not image.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="An image file is required.")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Upload an image file.")

    try:
        image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit.",
            )
        return image_bytes, preprocess_image(image_bytes)
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    finally:
        await image.close()


def prediction_details(scores: np.ndarray) -> tuple[int, PredictionResponse]:
    """Validate model scores and format the prediction response shared by both endpoints."""
    if scores.shape != (len(CLASS_NAMES),) or not np.all(np.isfinite(scores)):
        raise ValueError("Model returned an invalid prediction shape.")

    prediction_index = int(np.argmax(scores))
    return prediction_index, PredictionResponse(
        predicted_class=CLASS_NAMES[prediction_index],
        confidence=float(scores[prediction_index]),
        probabilities={name: float(score) for name, score in zip(CLASS_NAMES, scores)},
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(response: Response) -> HealthResponse:
    """Public readiness check."""
    if app.state.model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable")
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={
        400: {"description": "Missing image file."},
        413: {"description": "Uploaded file or decoded image is too large."},
        415: {"description": "Unsupported upload media type."},
        422: {"description": "Invalid, corrupted, or unsupported image format."},
        500: {"description": "Prediction failed."},
        503: {"description": "Model is unavailable."},
    },
)
async def predict(
    image: UploadFile | None = File(default=None, description="Chest X-ray image file"),
) -> PredictionResponse:
    """Classify a JPEG, PNG, BMP, or WEBP chest X-ray image."""
    _, batch = await read_and_preprocess_upload(image)

    model = get_loaded_model()
    try:
        probabilities = await run_in_threadpool(model.predict, batch, verbose=0)
        scores = np.asarray(probabilities[0], dtype=float)
        _, response = prediction_details(scores)
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try another image.",
        ) from exc

    return response


@app.post(
    "/gradcam",
    response_model=GradCAMResponse,
    tags=["Grad-CAM"],
    responses={
        400: {"description": "Missing image file."},
        413: {"description": "Uploaded file or decoded image is too large."},
        415: {"description": "Unsupported upload media type."},
        422: {"description": "Invalid, corrupted, or unsupported image format."},
        500: {"description": "Grad-CAM generation failed."},
        503: {"description": "Model is unavailable."},
    },
)
async def gradcam(
    image: UploadFile | None = File(default=None, description="Chest X-ray image file"),
) -> GradCAMResponse:
    """Classify an image and return a Grad-CAM PNG overlay for its predicted class."""
    image_bytes, batch = await read_and_preprocess_upload(image)
    model = get_loaded_model()
    try:
        result = await run_in_threadpool(generate_gradcam, model, batch, image_bytes)
        _, prediction = prediction_details(result.scores)
    except GradCAMError as exc:
        logger.exception("Grad-CAM generation failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grad-CAM visualization could not be generated. Please try another image.",
        ) from exc
    except Exception as exc:
        logger.exception("Grad-CAM prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grad-CAM visualization could not be generated. Please try another image.",
        ) from exc

    image_id = gradcam_image_store.put(result.image_bytes)
    return GradCAMResponse(
        **prediction.model_dump(),
        gradcam_image_url=f"/gradcam/image/{image_id}",
    )


@app.get(
    "/gradcam/image/{image_id}",
    response_class=Response,
    tags=["Grad-CAM"],
    summary="View a generated Grad-CAM PNG",
    description=(
        "Serves a generated Grad-CAM image for a limited time. The image is "
        f"held in memory for up to {GRADCAM_IMAGE_TTL_SECONDS // 60} minutes and is then removed."
    ),
    responses={
        200: {"description": "Grad-CAM PNG image."},
        404: {"description": "Image not found or expired."},
    },
)
def gradcam_image(image_id: str) -> Response:
    """Return a temporary Grad-CAM PNG so browsers can render it directly."""
    image_bytes = gradcam_image_store.get(image_id)
    if image_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Grad-CAM image was not found or has expired.",
        )
    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
