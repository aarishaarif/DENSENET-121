"""Lightweight API tests using a fake model, not the trained artifact."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import httpx
from PIL import Image

import app.main as main
from app.gradcam import GradCAMResult


class FakeModel:
    """Small deterministic substitute that keeps tests independent of TensorFlow inference."""

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert batch.shape == (1, 224, 224, 3)
        return np.array([[0.02, 0.03, 0.95]], dtype=np.float32)


def valid_image_bytes(image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture()
def api_app():
    main.app.state.model = FakeModel()
    yield main.app


def image_upload(content: bytes, content_type: str = "image/png") -> dict[str, tuple[str, bytes, str]]:
    return {"image": ("xray.png", content, content_type)}


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_root(api_app) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://testserver"
    ) as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Chest X-Ray Classification API is running",
        "docs": "/docs",
        "health": "/health",
    }


@pytest.mark.anyio
async def test_health_when_model_is_available(api_app) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://testserver"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_health_is_unavailable_without_a_model(api_app) -> None:
    main.app.state.model = None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://testserver"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


@pytest.mark.anyio
async def test_predict_rejects_invalid_image(api_app) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://testserver"
    ) as client:
        response = await client.post("/predict", files=image_upload(b"not an image"))
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_rejects_unsupported_media_type(api_app) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/predict",
            files=image_upload(b"plain text", content_type="text/plain"),
        )
    assert response.status_code == 415


@pytest.mark.anyio
async def test_predict_response_structure(api_app) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://testserver"
    ) as client:
        response = await client.post("/predict", files=image_upload(valid_image_bytes()))
    assert response.status_code == 200
    assert response.json() == {
        "predicted_class": "PNEUMONIA",
        "confidence": pytest.approx(0.95),
        "probabilities": {
            "COVID": pytest.approx(0.02),
            "NORMAL": pytest.approx(0.03),
            "PNEUMONIA": pytest.approx(0.95),
        },
    }


@pytest.mark.anyio
async def test_gradcam_response_structure(api_app, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate_gradcam(model, batch: np.ndarray, image_bytes: bytes) -> GradCAMResult:
        assert batch.shape == (1, 224, 224, 3)
        assert image_bytes
        return GradCAMResult(
            scores=np.array([0.02, 0.03, 0.95], dtype=float),
            image_bytes=b"fake-png",
        )

    monkeypatch.setattr(main, "generate_gradcam", fake_generate_gradcam)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://testserver"
    ) as client:
        response = await client.post("/gradcam", files=image_upload(valid_image_bytes()))

    assert response.status_code == 200
    assert response.json() == {
        "predicted_class": "PNEUMONIA",
        "confidence": pytest.approx(0.95),
        "probabilities": {
            "COVID": pytest.approx(0.02),
            "NORMAL": pytest.approx(0.03),
            "PNEUMONIA": pytest.approx(0.95),
        },
        "gradcam_image_url": response.json()["gradcam_image_url"],
    }
    image_url = response.json()["gradcam_image_url"]
    assert image_url.startswith("/gradcam/image/")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://testserver"
    ) as client:
        image_response = await client.get(image_url)
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"
    assert image_response.content == b"fake-png"


def test_openapi_has_no_security_scheme() -> None:
    schema = main.app.openapi()
    predict_operation = schema["paths"]["/predict"]["post"]
    gradcam_operation = schema["paths"]["/gradcam"]["post"]

    assert "security" not in predict_operation
    assert "security" not in gradcam_operation
    assert "securitySchemes" not in schema.get("components", {})
