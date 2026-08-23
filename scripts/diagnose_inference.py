"""Compare FastAPI preprocessing with the verified Kaggle inference pipeline.

Usage:
    .venv/bin/python scripts/diagnose_inference.py /path/to/PNEUMONIA_1563.png
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from app.model import MODEL_ARCHIVE, load_model
from app.preprocessing import IMAGE_SIZE, preprocess_image


def print_tensor_diagnostics(name: str, decoded: np.ndarray, tensor: np.ndarray) -> None:
    """Print every relevant stage needed to compare the two inference paths."""
    print(f"\n{name}")
    print(f"decoded shape: {decoded.shape}")
    print(f"decoded dtype: {decoded.dtype}")
    print(f"before preprocessing min/max: {decoded.min()} / {decoded.max()}")
    print(f"after DenseNet preprocessing min/max: {tensor.min()} / {tensor.max()}")
    print(f"final tensor shape: {tensor.shape}")
    print(f"final tensor dtype: {tensor.dtype}")


def fastapi_stages(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Mirror app.preprocessing.preprocess_image while retaining intermediate data."""
    with Image.open(BytesIO(image_bytes)) as source:
        decoded = np.asarray(
            source.convert("RGB").resize(IMAGE_SIZE, resample=Image.Resampling.NEAREST),
            dtype=np.float32,
        )
    return decoded, preprocess_image(image_bytes)


def kaggle_stages(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The verified Kaggle code, unchanged except for retaining intermediates."""
    img = keras_image.load_img(path, target_size=(224, 224), color_mode="rgb")
    decoded = keras_image.img_to_array(img)
    tensor = np.expand_dims(preprocess_input(decoded.copy()), axis=0)
    return decoded, tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the X-ray image to compare")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    image_bytes = args.image.read_bytes()
    api_decoded, api_tensor = fastapi_stages(image_bytes)
    kaggle_decoded, kaggle_tensor = kaggle_stages(args.image)
    print(f"Model archive: {MODEL_ARCHIVE}")
    print_tensor_diagnostics("A. FastAPI pipeline", api_decoded, api_tensor)
    print_tensor_diagnostics("B. Verified Kaggle-equivalent pipeline", kaggle_decoded, kaggle_tensor)
    print(f"\nDecoded arrays exactly equal: {np.array_equal(api_decoded, kaggle_decoded)}")
    print(f"Input tensors exactly equal: {np.array_equal(api_tensor, kaggle_tensor)}")
    print(f"Largest input-tensor difference: {np.max(np.abs(api_tensor - kaggle_tensor))}")

    model = load_model()
    api_probabilities = model.predict(api_tensor, verbose=0)[0]
    kaggle_probabilities = model.predict(kaggle_tensor, verbose=0)[0]
    print(f"\nA. FastAPI raw model probability array: {api_probabilities}")
    print(f"B. Kaggle raw model probability array: {kaggle_probabilities}")
    print(f"Largest probability difference: {np.max(np.abs(api_probabilities - kaggle_probabilities))}")


if __name__ == "__main__":
    main()
