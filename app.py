# -----------------------------
# Heart Disease Risk Prediction Flask API
# Updated for exam
# -----------------------------

from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Deployment folder and files
# -----------------------------
DEPLOYMENT_DIR = "deployment\"
MODEL_PATH = os.path.join(DEPLOYMENT_DIR, "model.pkl")
FEATURES_PATH = os.path.join(DEPLOYMENT_DIR, "feature_columns.txt")
CLASSES_PATH = os.path.join(DEPLOYMENT_DIR, "class_names.txt")

# -----------------------------
# Check if files exist
# -----------------------------
if not os.path.exists(DEPLOYMENT_DIR):
    raise FileNotFoundError(f"Deployment folder '{DEPLOYMENT_DIR}' not found!")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"Feature columns file '{FEATURES_PATH}' not found!")

if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Class names file '{CLASSES_PATH}' not found!")

# -----------------------------
# Load model and metadata
# -----------------------------
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_columns = [line.strip() for line in f.readlines()]

with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# Home endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Heart Disease Risk Prediction API is running",
        "model_loaded": True,
        "num_features": len(feature_columns),
        "classes": class_names
    })

# Prediction endpoint
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        # Check for missing features
        missing_features = [f for f in feature_columns if f not in data]
        if missing_features:
            return jsonify({
                "error": "Missing input features",
                "missing_features": missing_features
            }), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data], columns=feature_columns)

        # Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))

        result = {
            "predicted_class": int(prediction),
            "predicted_label": class_names[int(prediction)],
            "confidence": round(confidence, 4),
            "class_probabilities": {
                class_names[i]: round(float(probabilities[i]), 4)
                for i in range(len(class_names))
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
