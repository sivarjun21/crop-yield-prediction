"""
app.py
------
Flask web application. It does NOT train anything: it loads the neural network
and preprocessor that train.py already saved in /models.

Routes
    GET  /              the prediction page
    GET  /performance   model performance page (metrics, graphs, architecture)
    GET  /api/options   dropdown values + valid ranges for the frontend
    POST /predict       JSON in -> predicted yield out
    GET  /health        quick "is the model loaded?" check
    GET  /results/<f>   the graphs saved by evaluate.py

Run:   python app.py      then open   http://127.0.0.1:5000
"""

import json
import math
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, jsonify, render_template, request, send_from_directory
from tensorflow import keras

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.keras"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
RESULTS_DIR = BASE_DIR / "results"

HG_PER_TONNE = 10_000  # 1 tonne = 10,000 hectograms

# Hard limits: values outside these are physically impossible or absurd -> rejected.
LIMITS = {
    "rainfall": (0, 12000),        # mm/year   (wettest places on Earth are ~11,900 mm)
    "pesticides": (0, 10_000_000), # tonnes
    "temperature": (-50, 60),      # degrees Celsius
    "year": (1900, 2100),
}

app = Flask(__name__)


# ----------------------------------------------------------------------------
# Load the trained artifacts once, when the server starts
# ----------------------------------------------------------------------------
def load_artifacts():
    """Return (model, preprocessor, metadata, error_message)."""
    try:
        if not (MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and METADATA_PATH.exists()):
            return None, None, None, "The trained model was not found. Run  python train.py  first."
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        trained_with = meta.get("versions", {}).get("scikit_learn")
        if trained_with and trained_with != sklearn.__version__:
            print(f"[warning] preprocessor.pkl was created with scikit-learn {trained_with}, "
                  f"but {sklearn.__version__} is installed. If loading fails, run  python train.py  again.")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = keras.models.load_model(MODEL_PATH)
        return model, preprocessor, meta, None
    except Exception as exc:  # noqa: BLE001 - show a friendly message, log the details
        print(f"[error] could not load the model: {exc!r}")
        return None, None, None, (
            "The saved model could not be loaded (usually a library version mismatch). "
            "Install the versions in requirements.txt and run  python train.py  again."
        )


MODEL, PREPROCESSOR, META, LOAD_ERROR = load_artifacts()

if META:
    COUNTRY_LOOKUP = {c.lower(): c for c in META["countries"]}
    CROP_LOOKUP = {c.lower(): c for c in META["crops"]}
    KNOWN_PAIRS = set(META["country_crop_pairs"])
else:
    COUNTRY_LOOKUP, CROP_LOOKUP, KNOWN_PAIRS = {}, {}, set()


# ----------------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------------
class InputError(ValueError):
    """A problem with the user's input. Carries the field name for the UI."""

    def __init__(self, message, field=None):
        super().__init__(message)
        self.field = field


def parse_number(data, field, label, integer=False):
    raw = data.get(field)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise InputError(f"{label} is required.", field)
    if isinstance(raw, bool):
        raise InputError(f"{label} must be a number.", field)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise InputError(f"{label} must be a number.", field) from None
    if math.isnan(value) or math.isinf(value):
        raise InputError(f"{label} must be a valid number.", field)
    if integer and value != int(value):
        raise InputError(f"{label} must be a whole number.", field)
    low, high = LIMITS[field]
    if field in ("rainfall", "pesticides") and value < 0:
        raise InputError(f"{label} cannot be negative.", field)
    if not low <= value <= high:
        raise InputError(f"{label} must be between {low:,} and {high:,}.", field)
    return int(value) if integer else value


def parse_choice(data, field, label, lookup):
    raw = data.get(field)
    if not isinstance(raw, str) or not raw.strip():
        raise InputError(f"Please choose a {label.lower()}.", field)
    match = lookup.get(raw.strip().lower())
    if match is None:
        raise InputError(f"'{raw.strip()}' is not in the training data. Pick a {label.lower()} from the list.", field)
    return match


def range_warnings(values):
    """Soft warnings: allowed input, but outside what the network learned from."""
    warnings = []
    names = {"year": "Year", "rainfall": "Rainfall", "pesticides": "Pesticides", "temperature": "Temperature"}
    units = {"year": "", "rainfall": " mm", "pesticides": " tonnes", "temperature": " °C"}
    for key in ("year", "rainfall", "pesticides", "temperature"):
        low, high = META["ranges"][key]["min"], META["ranges"][key]["max"]
        if not low <= values[key] <= high:
            warnings.append(
                f"{names[key]} is outside the range seen in training data "
                f"({low:.6g}–{high:.6g}{units[key]}), so the estimate is less reliable."
            )
    if f"{values['country']}|{values['crop']}" not in KNOWN_PAIRS:
        warnings.append(
            f"The training data has no {values['crop']} records for {values['country']}, "
            "so the estimate is less reliable."
        )
    return warnings


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/performance")
def performance():
    summary = None
    if META and "metrics" in META:
        history = META["history"]
        best = int(np.argmin(history["val_loss"]))  # epoch with the lowest validation loss
        summary = {
            "best_epoch": best + 1,
            "train_loss": history["loss"][best],
            "val_loss": history["val_loss"][best],
        }
    return render_template("performance.html", meta=META, summary=summary, error=LOAD_ERROR)


@app.get("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.get("/health")
def health():
    return jsonify({"model_loaded": MODEL is not None, "error": LOAD_ERROR})


@app.get("/api/options")
def options():
    if META is None:
        return jsonify({"error": LOAD_ERROR}), 503
    return jsonify(
        {
            "countries": META["countries"],
            "crops": META["crops"],
            "ranges": META["ranges"],
            "examples": META.get("examples", [])[:5],
            "dataset": {
                "rows": META["cleaning_report"]["clean_rows"],
                "countries": len(META["countries"]),
                "crops": len(META["crops"]),
            },
        }
    )


@app.post("/predict")
def predict():
    if MODEL is None:
        return jsonify({"error": LOAD_ERROR or "The model is not available."}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send the inputs as a JSON object."}), 400

    try:
        values = {
            "country": parse_choice(data, "country", "Country", COUNTRY_LOOKUP),
            "crop": parse_choice(data, "crop", "Crop", CROP_LOOKUP),
            "year": parse_number(data, "year", "Year", integer=True),
            "rainfall": parse_number(data, "rainfall", "Rainfall"),
            "pesticides": parse_number(data, "pesticides", "Pesticides"),
            "temperature": parse_number(data, "temperature", "Temperature"),
        }
    except InputError as err:
        return jsonify({"error": str(err), "field": err.field}), 400

    try:
        # Same column names/order as during training
        row = pd.DataFrame([values])[["country", "crop", "year", "rainfall", "pesticides", "temperature"]]
        z = float(MODEL.predict(PREPROCESSOR.transform(row), verbose=0).ravel()[0])

        # network output (standardised) -> hg/ha -> tonnes/ha
        hg_per_ha = z * META["target_std"] + META["target_mean"]
        # A yield can never be below zero; this only affects extreme out-of-range inputs.
        hg_per_ha = max(hg_per_ha, 0.0)
        tonnes_per_ha = hg_per_ha / HG_PER_TONNE

        quantiles = META["crop_quantiles"][values["crop"]]
        percentile = float(np.interp(tonnes_per_ha, quantiles, np.linspace(0, 100, len(quantiles))))
    except Exception as exc:  # noqa: BLE001
        app.logger.error("Prediction failed: %r", exc)
        return jsonify({"error": "The prediction could not be completed. Please check the inputs and try again."}), 500

    return jsonify(
        {
            "predicted_yield_tonnes_per_hectare": round(tonnes_per_ha, 2),
            "predicted_yield_kg_per_hectare": round(tonnes_per_ha * 1000),
            "predicted_yield_hg_per_hectare": round(hg_per_ha),
            "inputs": values,
            "warnings": range_warnings(values),
            "context": {
                "crop": values["crop"],
                "percentile": round(percentile, 1),
                "lowest": quantiles[0],
                "median": quantiles[50],
                "highest": quantiles[-1],
            },
            "note": "AI/Deep Learning estimate based on historical training data, not a guaranteed outcome.",
        }
    )


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Page not found."}), 404


@app.errorhandler(500)
def server_error(_):
    return jsonify({"error": "Something went wrong on the server."}), 500


if __name__ == "__main__":
    if LOAD_ERROR:
        print(f"\n[!] {LOAD_ERROR}\n")
    # debug=False: the reloader would load TensorFlow twice and never shows stack traces to users
    app.run(host="127.0.0.1", port=5000, debug=False)
