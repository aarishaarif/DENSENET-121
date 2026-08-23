# Chest X-Ray Classification API

FastAPI service for the supplied DenseNet121 3-class chest X-ray model.

## Project structure

```text
app/                  FastAPI routes, model loading, schemas, and preprocessing
model/best_model.zip  Supplied Keras config and weights archive
scripts/              Local inference diagnostic utility
tests/                Lightweight API tests using a fake model
```

## Model storage and loading

`model/best_model.zip` contains a Keras 3 `Functional` model configuration in
`config.json` and its parameters in `model.weights.h5` (plus metadata). The
configuration is the source of truth: it describes the DenseNet121 graph,
followed by global average pooling, `Dense(256, relu)`, `Dropout(0.4)`, and a
3-output softmax layer. At startup, the API restores that exact graph with
`keras.models.model_from_json(config_json)`, temporarily extracts the H5 file,
and calls `model.load_weights(...)`. It does not retrain, replace, or otherwise
alter the architecture.

Images are converted to RGB, resized to 224×224, converted to a NumPy array,
processed with `tensorflow.keras.applications.densenet.preprocess_input`, and
given a batch dimension. Output indices map to `0=COVID`, `1=NORMAL`, and
`2=PNEUMONIA`.

The resize explicitly uses nearest-neighbor interpolation because that is the
default used by Keras `load_img(..., target_size=(224, 224))`.

## Setup and run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open Swagger UI at `http://127.0.0.1:8000/docs`.

## Endpoints

- `GET /` — public API landing response.
- `GET /health` — public readiness check. Returns `200` when the model is
  loaded and `503` when it is unavailable; it does not run inference.
- `POST /predict` — classification endpoint.
- `POST /gradcam` — classification plus an in-memory Grad-CAM visualization.

## Prediction request

`POST /predict` accepts multipart form data with the file field named `image`.
Supported formats are JPEG, PNG, BMP, and WEBP. Uploads are limited to 10 MiB
and decoded images to 25,000,000 pixels.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F 'image=@chest-xray.jpg'
```

Example response:

```json
{
  "predicted_class": "PNEUMONIA",
  "confidence": 0.94,
  "probabilities": {"COVID": 0.02, "NORMAL": 0.04, "PNEUMONIA": 0.94}
}
```

Common errors: `400` for a missing image, `413` for an oversized upload/image,
`415` for a non-image upload, `422` for invalid or corrupted image data, and
`503` when the model is unavailable.

## Grad-CAM visualization

`POST /gradcam` accepts the same `image` upload as `/predict`. It uses the
model's predicted class automatically and returns the ordinary prediction
fields plus a temporary `gradcam_image_url`, for example
`/gradcam/image/<unique-id>`. Open that URL in a browser to view the PNG
overlay directly. The generated image is held in memory for up to 10 minutes,
is bounded to 256 stored images, and is never saved permanently to disk.

In Swagger, open `POST /gradcam`, select **Try it out**, upload an X-ray in the
`image` field, and click **Execute**. Copy the returned `gradcam_image_url`
into a new browser tab (using the same API host) to display the image. The
`GET /gradcam/image/{image_id}` endpoint serves it with `image/png`; expired
or unknown IDs return `404`. The returned overlay has the original uploaded
image dimensions.

The reconstructed model is a flat Functional graph, not a nested DenseNet
model. Grad-CAM programmatically selects its deepest accessible `Conv2D` layer:
`conv5_block16_2_conv` (a 7×7×32 convolution). This layer feeds the final
DenseNet feature concatenation and is the final convolutional activation before
the classifier head.

## Compare inference preprocessing

To compare an uploaded-file-equivalent FastAPI tensor with the verified Kaggle
pipeline for the same image, run:

```bash
.venv/bin/python scripts/diagnose_inference.py /path/to/PNEUMONIA_1563.png
```

It prints decoded shapes, dtypes, value ranges before and after DenseNet
preprocessing, final tensor details, and raw probabilities from both paths.

## Tests

Install test dependencies and run the lightweight suite:

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake model and do not retrain, alter, or run inference against
the supplied model artifact.
