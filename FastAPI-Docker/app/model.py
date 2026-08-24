"""Loading the archived Keras configuration and its trained weights."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ARCHIVE = PROJECT_ROOT / "model" / "best_model.zip"


class ModelLoadError(RuntimeError):
    """Raised when the serialized model cannot be reconstructed."""


def load_model() -> keras.Model:
    """Rebuild the exact Functional graph in config.json and load its H5 weights."""
    if not MODEL_ARCHIVE.is_file():
        raise ModelLoadError(f"Model archive was not found: {MODEL_ARCHIVE}")

    try:
        with zipfile.ZipFile(MODEL_ARCHIVE) as archive:
            required = {"config.json", "model.weights.h5"}
            missing = required.difference(archive.namelist())
            if missing:
                raise ModelLoadError(
                    f"Model archive is missing required file(s): {', '.join(sorted(missing))}"
                )

            # model_from_json restores the serialized Functional graph exactly;
            # weights are temporarily materialized because Keras expects an H5 path.
            config_json = archive.read("config.json").decode("utf-8")
            model = keras.models.model_from_json(config_json)
            with tempfile.TemporaryDirectory(prefix="chest-xray-weights-") as directory:
                weights_path = Path(directory) / "model.weights.h5"
                weights_path.write_bytes(archive.read("model.weights.h5"))
                model.load_weights(weights_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ModelLoadError("Could not reconstruct the trained model from its archive.") from exc

    return model
