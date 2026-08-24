"""Image validation and preprocessing matching the model training pipeline."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.densenet import preprocess_input

IMAGE_SIZE = (224, 224)
# The API only needs a 224×224 result. These limits prevent unusually large
# uploads/decoded images from consuming excessive memory before resizing.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB
MAX_IMAGE_PIXELS = 25_000_000


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a supported, readable image."""


class ImageTooLargeError(InvalidImageError):
    """Raised when an upload would exceed safe resource limits."""


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Return a batched DenseNet121-ready image tensor."""
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty.")

    try:
        # verify() detects truncated/corrupted streams before decoding them.
        with Image.open(BytesIO(image_bytes)) as source:
            image_format = source.format
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ImageTooLargeError(
                    f"Image is too large. Maximum decoded size is {MAX_IMAGE_PIXELS:,} pixels."
                )
            source.verify()

        if image_format not in {"JPEG", "PNG", "BMP", "WEBP"}:
            raise InvalidImageError(
                "Unsupported image format. Upload a JPEG, PNG, BMP, or WEBP image."
            )

        with Image.open(BytesIO(image_bytes)) as source:
            # Keras load_img(..., target_size=(224, 224)) defaults to nearest
            # interpolation. Specify it explicitly so API and Kaggle tensors match.
            image = source.convert("RGB").resize(IMAGE_SIZE, resample=Image.Resampling.NEAREST)
            array = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, InvalidImageError):
            raise
        raise InvalidImageError("The uploaded file is not a valid, readable image.") from exc

    # This is the exact DenseNet preprocessing used by training.
    return np.expand_dims(preprocess_input(array), axis=0)
