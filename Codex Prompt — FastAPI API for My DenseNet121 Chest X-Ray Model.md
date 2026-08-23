I want you to build a FastAPI REST API for my existing chest X-ray classification model.

IMPORTANT:
Do NOT start coding immediately.

First inspect the entire project and especially:

models/best_model.zip

The ZIP contains:

- metadata.json
- config.json
- model.weights.h5

The model was trained for 3-class chest X-ray classification:

0 = COVID
1 = NORMAL
2 = PNEUMONIA

Input shape:

224 x 224 x 3

Preprocessing:

tensorflow.keras.applications.densenet.preprocess_input

The model is based on DenseNet121 transfer learning.

DO NOT retrain the model.

DO NOT change the model architecture.

DO NOT assume that best_model.keras exists.

First inspect config.json inside best_model.zip and determine the exact Keras architecture stored there.

Then determine the correct way to reconstruct the Keras model from:

config.json
+
model.weights.h5

The API must use the EXACT trained architecture and weights.

After inspecting the model, explain to me briefly:

1. How the model is stored
2. How you will reconstruct/load it
3. How you will load model.weights.h5
4. What preprocessing will be used
5. What the class mapping is

Then implement the FastAPI REST API.

Create:

app/main.py
app/model.py
app/preprocessing.py
app/schemas.py

and:

requirements.txt
.env
.gitignore
README.md

API endpoints:

GET /health

POST /predict

/predict must accept an uploaded X-ray image using multipart/form-data.

The response should be:

{
    "predicted_class": "PNEUMONIA",
    "confidence": 0.94,
    "probabilities": {
        "COVID": 0.02,
        "NORMAL": 0.04,
        "PNEUMONIA": 0.94
    }
}

These numbers are only an example. Use actual model predictions.

Preprocessing MUST exactly match training:

1. Load image
2. Convert to RGB
3. Resize to 224x224
4. Convert to NumPy array
5. Apply DenseNet preprocess_input
6. Add batch dimension
7. Run model.predict()

Do NOT use StandardScaler or MinMaxScaler.

Load the model only once when FastAPI starts.

Add API-key authentication to /predict.

Use:

API_KEY

from .env/environment variable.

Never hardcode the API key.

Keep /health public.

Add Swagger documentation at:

/docs

Add proper error handling for:

- missing image
- invalid image
- corrupted image
- unsupported image
- model loading failure
- prediction failure

Create a requirements.txt with the actual dependencies required.

Use:

uvicorn app.main:app --reload

for local development.

Also create a Dockerfile later, but FIRST make the API work locally.

IMPORTANT:

Do not create a fake model.
Do not create dummy predictions.
Do not retrain anything.
Do not modify the trained architecture.
Do not guess the architecture.

The actual source of truth is:

models/best_model.zip

and its config.json + model.weights.h5.

Before proceeding, inspect the ZIP and show me what you found.

