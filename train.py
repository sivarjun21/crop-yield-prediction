"""
train.py
--------
Trains the crop-yield neural network ONCE and saves everything the web app needs.

Pipeline (in this exact order, so no test data leaks into training):

    1. Load + clean the dataset
    2. Split into train / validation / test
    3. Fit the preprocessor on TRAINING data only
    4. Transform train / validation / test with that fitted preprocessor
    5. Build the neural network (MLP)
    6. Train with EarlyStopping + ModelCheckpoint
    7. Save model (.keras), preprocessor (.pkl) and metadata (.json)
    8. Evaluate on the test set and draw the graphs (evaluate.py)

Run:   python train.py
"""

import json
import os
import platform
import random
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's very chatty logs

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_utils import (
    CATEGORICAL,
    HG_PER_TONNE,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET,
    build_preprocessor,
    load_and_clean,
    split_data,
)
from evaluate import run_evaluation

MAX_EPOCHS = 300
BATCH_SIZE = 64


def set_seeds(seed=RANDOM_STATE):
    """Fix every random number generator so results are repeatable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_model(n_features):
    """Input -> Dense128 -> Dropout -> Dense64 -> Dropout -> Dense32 -> Dense1 (linear)."""
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear"),  # regression output
        ],
        name="crop_yield_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model


def main():
    started = time.time()
    set_seeds()
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Load + clean ---------------------------------------------------------
    df, report = load_and_clean()
    print(f"[1/8] Dataset: {report['raw_rows']:,} raw rows -> {report['clean_rows']:,} clean rows")
    print(f"      duplicates removed: {report['duplicate_rows_removed']:,} | "
          f"rows with missing values: {report['rows_with_missing_values']} | "
          f"impossible rows removed: {report['impossible_rows_removed']}")

    # 2. Split BEFORE fitting anything ----------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"[2/8] Split: train={len(X_train):,}  validation={len(X_val):,}  test={len(X_test):,}")

    # 3 + 4. Fit preprocessing on TRAIN only, then transform every split ------
    preprocessor = build_preprocessor()
    Xt_train = preprocessor.fit_transform(X_train)   # fit ONLY here
    Xt_val = preprocessor.transform(X_val)           # transform only
    Xt_test = preprocessor.transform(X_test)         # transform only
    print(f"[3/8] Preprocessing fitted on training data: {Xt_train.shape[1]} input features "
          f"(country + crop one-hot columns and 4 numeric columns)")

    # Standardise the target for stable training (mean/std from TRAIN only).
    # The web app converts the network output back to hg/ha and then to t/ha.
    target_mean = float(y_train.mean())
    target_std = float(y_train.std())
    z_train = ((y_train - target_mean) / target_std).values
    z_val = ((y_val - target_mean) / target_std).values

    # 5. Model ------------------------------------------------------------------
    model = build_model(Xt_train.shape[1])
    model.summary()

    # 6. Train ------------------------------------------------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]
    print("[6/8] Training ...")
    history = model.fit(
        Xt_train, z_train,
        validation_data=(Xt_val, z_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # 7. Save ---------------------------------------------------------------------
    model.save(MODEL_PATH)                 # best weights (restored by EarlyStopping)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    cat = preprocessor.named_transformers_["categories"]
    countries = sorted(str(c) for c in cat.categories_[0])
    crops = sorted(str(c) for c in cat.categories_[1])

    # Ranges of the TRAINING data (used by the app to warn about out-of-range input)
    ranges = {
        col: {"min": float(X_train[col].min()), "max": float(X_train[col].max())}
        for col in ["year", "rainfall", "pesticides", "temperature"]
    }
    # Which country/crop pairs actually exist in the training data
    combos = sorted(set(map(tuple, X_train[CATEGORICAL].drop_duplicates().values.tolist())))
    # Recorded yield distribution per crop (t/ha), 101 quantiles -> used for "percentile" context
    quantiles = np.linspace(0, 1, 101)
    crop_quantiles = {
        crop: [round(float(v), 4) for v in (grp[TARGET] / HG_PER_TONNE).quantile(quantiles)]
        for crop, grp in df.groupby("crop")
    }

    meta = {
        "project": "Deep Learning Based Crop Yield Prediction System",
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "target_unit_in_dataset": "hg/ha",
        "hg_per_tonne": HG_PER_TONNE,
        "target_mean": target_mean,
        "target_std": target_std,
        "countries": countries,
        "crops": crops,
        "ranges": ranges,
        "country_crop_pairs": [f"{c}|{k}" for c, k in combos],
        "crop_quantiles": crop_quantiles,
        "cleaning_report": report,
        "training": {
            "n_input_features": int(Xt_train.shape[1]),
            "parameters": int(model.count_params()),
            "epochs_run": len(history.history["loss"]),
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam (learning rate 0.001)",
            "loss": "Mean Squared Error",
            "random_state": RANDOM_STATE,
            "seconds": round(time.time() - started, 1),
        },
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[7/8] Saved: {MODEL_PATH.name}, {PREPROCESSOR_PATH.name}, {METADATA_PATH.name} in /models")

    # 8. Evaluate on the untouched test set ---------------------------------------
    print("[8/8] Evaluating on the test set ...\n")
    meta = run_evaluation(model, preprocessor, meta)
    print("\nDone. Now start the web app with:  python app.py")
    return meta


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
