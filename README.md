# Chest X-Ray Classification API

An end-to-end **chest X-ray classification project** using a **DenseNet121-based deep learning model**, **Grad-CAM explainability**, **FastAPI**, and **Docker**.

The project provides a REST API that accepts chest X-ray images and returns a predicted class, confidence score, class probabilities, and an optional Grad-CAM visualization.
V

---

## 🚀 Features

- DenseNet121-based 3-class chest X-ray classification
- Keras 3 model reconstruction from a saved model configuration
- DenseNet-specific image preprocessing
- FastAPI inference API
- `/predict` classification endpoint
- `/gradcam` explainability endpoint
- Grad-CAM heatmap generation
- Health/readiness endpoint
- Input validation and upload-size limits
- Lightweight automated API tests
- Dockerized API
- Local inference diagnostic utility

---

## 🧠 Model

The supplied model is a Keras 3 `Functional` model built around a DenseNet121 feature extractor.

The classifier head is:

```text
DenseNet121
     ↓
Global Average Pooling
     ↓
Dense(256, ReLU)
     ↓
Dropout(0.4)
     ↓
Dense(3, Softmax)
```

### Input

```text
224 × 224 × 3
```

### Output classes

```text
0 → COVID
1 → NORMAL
2 → PNEUMONIA
```

The model artifact is stored at:

```text
model/best_model.zip
```

---

## 📦 Model Storage and Loading

`model/best_model.zip` contains the Keras model configuration and weights:

```text
config.json
model.weights.h5
metadata
```

The configuration is the source of truth for the model architecture. At application startup, the API:

1. Reads the model configuration.
2. Reconstructs the Functional model using `keras.models.model_from_json(config_json)`.
3. Temporarily extracts the H5 weights file.
4. Loads the weights using `model.load_weights(...)`.

The API does not retrain or replace the supplied model architecture.

---

## 🖼️ Inference Preprocessing

Uploaded images are processed as follows:

```text
Uploaded Image
      ↓
Convert to RGB
      ↓
Resize to 224 × 224
      ↓
Convert to NumPy array
      ↓
DenseNet preprocess_input
      ↓
Add batch dimension
      ↓
Model inference
```

The API uses:

```python
tensorflow.keras.applications.densenet.preprocess_input
```

The resize uses nearest-neighbor interpolation because this matches the default used by the Keras `load_img(..., target_size=(224, 224))` pipeline used for the verified inference comparison.

---

## 📊 Dataset and Training

The model was developed for a three-class chest X-ray classification task:

- COVID
- NORMAL
- PNEUMONIA

The training notebook contains the dataset preparation, image generators, augmentation, class-weight calculation, DenseNet121 model construction, training, fine-tuning, evaluation, and Grad-CAM analysis.

The model development workflow is:

```text
Chest X-Ray Dataset
        ↓
Data Preparation
        ↓
Image Augmentation
        ↓
Class Weight Calculation
        ↓
DenseNet121 Transfer Learning
        ↓
Fine-Tuning
        ↓
Model Evaluation
        ↓
Grad-CAM
        ↓
Saved Model Artifact
        ↓
FastAPI
```

### Evaluation

The notebook reports a validation accuracy of approximately:

```text
95.69%
```

with a macro F1-score of approximately:

```text
0.96
```

The validation evaluation contains:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| COVID | 0.96 | 1.00 | 0.98 |
| NORMAL | 0.93 | 0.97 | 0.95 |
| PNEUMONIA | 0.99 | 0.91 | 0.94 |

> These are validation results from the project workflow. They should not be interpreted as evidence of clinical performance or generalization to external clinical datasets.

---

## 🔍 Grad-CAM Explainability

The API includes **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize image regions associated with the model's prediction.

The `/gradcam` endpoint:

1. Accepts an X-ray image.
2. Applies the same preprocessing used for inference.
3. Runs model prediction.
4. Automatically selects the predicted class.
5. Computes Grad-CAM.
6. Generates a heatmap overlay.
7. Returns a temporary URL for the generated PNG.

### Grad-CAM layer

The reconstructed model is a flat Keras Functional graph rather than a nested DenseNet model.

The implementation programmatically selects the deepest accessible `Conv2D` layer:

```text
conv5_block16_2_conv
```

This is a `7 × 7 × 32` convolutional layer and is the final convolutional activation before the classifier head in the reconstructed graph.

### Temporary image storage

Generated Grad-CAM images:

- are stored in memory
- are not permanently written to disk
- expire after 10 minutes
- are limited to 256 stored images

---

# 🌐 FastAPI

The trained model is exposed through a REST API built with **FastAPI**.

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Public API landing response |
| `GET` | `/health` | Model readiness check |
| `POST` | `/predict` | Chest X-ray classification |
| `POST` | `/gradcam` | Classification + Grad-CAM |
| `GET` | `/gradcam/image/{image_id}` | Retrieve a generated Grad-CAM image |

---

## `/health`

Check whether the model is loaded:

```bash
curl http://127.0.0.1:8000/health
```

When available:

```json
{
  "status": "ok"
}
```

The endpoint returns `503` when the model is unavailable and does not run inference.

---

## `/predict`

`POST /predict` accepts multipart form data with the file field named:

```text
image
```

Supported image formats:

- JPEG
- PNG
- BMP
- WEBP

Uploads are limited to:

```text
10 MiB
```

Decoded images are limited to:

```text
25,000,000 pixels
```

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict   -F "image=@chest-xray.jpg"
```

### Example response

```json
{
  "predicted_class": "PNEUMONIA",
  "confidence": 0.94,
  "probabilities": {
    "COVID": 0.02,
    "NORMAL": 0.04,
    "PNEUMONIA": 0.94
  }
}
```

### Common errors

| Status | Meaning |
|---|---|
| `400` | Missing image |
| `413` | Oversized upload/image |
| `415` | Unsupported media type |
| `422` | Invalid or corrupted image |
| `503` | Model unavailable |

---

# 🔬 `/gradcam`

The `/gradcam` endpoint accepts the same `image` upload as `/predict`.

Example response:

```json
{
  "predicted_class": "PNEUMONIA",
  "confidence": 0.94,
  "probabilities": {
    "COVID": 0.02,
    "NORMAL": 0.04,
    "PNEUMONIA": 0.94
  },
  "gradcam_image_url": "/gradcam/image/<unique-id>"
}
```

Open the returned URL using the same API host to view the PNG overlay.

The `GET /gradcam/image/{image_id}` endpoint serves the generated visualization as `image/png`. Expired or unknown IDs return `404`.

The returned overlay uses the original uploaded image dimensions.

### Using Grad-CAM through Swagger

Open:

```text
http://127.0.0.1:8000/docs
```

Then:

1. Open `POST /gradcam`.
2. Click **Try it out**.
3. Upload an X-ray in the `image` field.
4. Click **Execute**.
5. Copy the returned `gradcam_image_url`.
6. Open the URL in a browser using the same API host.

---

# 🐳 Docker

The API is containerized using Docker.

The repository includes:

```text
Dockerfile
.dockerignore
```

## Build the image

From the project root:

```bash
docker build -t chest-xray-api .
```

## Run the container

```bash
docker run -p 8000:8000 chest-xray-api
```

The API will then be available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# 💻 Local Setup

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🧪 Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run:

```bash
pytest
```

The API tests use a **fake model** and do not retrain, modify, or run inference against the supplied model artifact.

---

# 🔬 Compare Inference Preprocessing

The repository contains a diagnostic utility for comparing an uploaded-file-equivalent FastAPI tensor with the verified Kaggle inference pipeline for the same image.

Run:

```bash
.venv/bin/python scripts/diagnose_inference.py /path/to/PNEUMONIA_1563.png
```

It reports:

- decoded image shapes
- data types
- value ranges before preprocessing
- value ranges after DenseNet preprocessing
- final tensor details
- raw probabilities from both inference paths

---

# 📁 Project Structure

```text
DENSENET-121/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── gradcam.py
│   └── schemas.py
│
├── model/
│   └── best_model.zip
│
├── scripts/
│   └── diagnose_inference.py
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

# 🛠️ Technologies

- Python
- TensorFlow / Keras
- DenseNet121
- FastAPI
- Uvicorn
- Pydantic
- NumPy
- Pillow
- scikit-learn
- Matplotlib
- Seaborn
- Docker
- Pytest
- Grad-CAM

---



## ⭐ Project Workflow

```text
Chest X-Ray Dataset
        │
        ▼
Preprocessing & Augmentation
        │
        ▼
DenseNet121
        │
        ▼
Transfer Learning + Fine-Tuning
        │
        ▼
Model Evaluation
        │
   ┌────┴────┐
   ▼         ▼
Prediction  Grad-CAM
   │         │
   └────┬────┘
        ▼
     FastAPI
        │
        ▼
      Docker
        │
        ▼
   API Service
```
