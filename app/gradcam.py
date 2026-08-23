"""Grad-CAM generation for the reconstructed DenseNet121 classifier."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


class GradCAMError(RuntimeError):
    """Raised when a Grad-CAM visualization cannot be generated."""


@dataclass(frozen=True)
class GradCAMResult:
    """Model scores and an in-memory PNG overlay returned by Grad-CAM."""

    scores: np.ndarray
    image_bytes: bytes


def find_last_convolutional_layer(model: keras.Model) -> keras.layers.Conv2D:
    """Return the deepest directly accessible 2D convolution in this model graph."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer
    raise GradCAMError("The loaded model has no accessible Conv2D layer for Grad-CAM.")


def _decode_original_rgb(image_bytes: bytes) -> Image.Image:
    """Decode a previously validated upload without altering its original size."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            return source.convert("RGB").copy()
    except (OSError, ValueError) as exc:
        raise GradCAMError("Could not decode the image for Grad-CAM visualization.") from exc


def _heatmap_overlay(heatmap: np.ndarray, original_image: Image.Image) -> bytes:
    """Resize, colorize, and blend a normalized heatmap over the original image."""
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L")
    heatmap_image = heatmap_image.resize(original_image.size, resample=Image.Resampling.BILINEAR)

    intensity = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    # A compact red-yellow colormap implemented without an additional plotting dependency.
    colored_heatmap = np.stack(
        (intensity, np.minimum(2.0 * intensity, 1.0), np.zeros_like(intensity)), axis=-1
    )
    original = np.asarray(original_image, dtype=np.float32) / 255.0
    overlay = np.clip(0.60 * original + 0.40 * colored_heatmap, 0, 1)

    output = BytesIO()
    Image.fromarray(np.uint8(overlay * 255), mode="RGB").save(output, format="PNG")
    return output.getvalue()


def generate_gradcam(model: keras.Model, batch: np.ndarray, image_bytes: bytes) -> GradCAMResult:
    """Compute Grad-CAM for the model's actual predicted class and return a PNG overlay."""
    try:
        convolutional_layer = find_last_convolutional_layer(model)
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[convolutional_layer.output, model.output],
        )

        image_tensor = tf.convert_to_tensor(batch)
        with tf.GradientTape() as tape:
            convolutional_output, predictions = grad_model(image_tensor, training=False)
            predicted_index = tf.argmax(predictions[0])
            predicted_score = predictions[:, predicted_index]

        gradients = tape.gradient(predicted_score, convolutional_output)
        if gradients is None:
            raise GradCAMError("Could not calculate gradients for Grad-CAM.")

        channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(convolutional_output[0] * channel_weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))

        scores = np.asarray(predictions[0].numpy(), dtype=float)
        heatmap_array = np.asarray(heatmap.numpy(), dtype=np.float32)
        return GradCAMResult(
            scores=scores,
            image_bytes=_heatmap_overlay(heatmap_array, _decode_original_rgb(image_bytes)),
        )
    except GradCAMError:
        raise
    except Exception as exc:
        raise GradCAMError("Grad-CAM visualization could not be generated.") from exc
