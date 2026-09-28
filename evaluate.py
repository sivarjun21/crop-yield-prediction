"""
evaluate.py
-----------
Evaluates the trained neural network on the TEST set (data the network never
saw during training) and saves:

    results/model_metrics.txt        MAE, MSE, RMSE, R2 (train + test)
    results/training_loss.png        training vs validation loss
    results/actual_vs_predicted.png  actual vs predicted yield (test set)
    results/residuals.png            prediction error distribution (test set)
    results/sample_predictions.txt   example test cases taken from real dataset rows

train.py calls this automatically at the end of training.
You can also run it on its own:   python evaluate.py
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files only (no pop-up window, works on any Windows setup)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import (
    HG_PER_TONNE,
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RESULTS_DIR,
    load_and_clean,
    split_data,
)

GREEN, AMBER, INK, GRID = "#1F6B45", "#C98A1B", "#1D2B23", "#DDE5DC"


def predict_hg_ha(model, preprocessor, meta, X):
    """features -> preprocessing -> network -> un-standardise -> hg/ha."""
    z = model.predict(preprocessor.transform(X), verbose=0).ravel()
    return z * meta["target_std"] + meta["target_mean"]


def score(y_true_hg, y_pred_hg):
    """Return metrics in tonnes/hectare (easier to read than hg/ha)."""
    y_true, y_pred = y_true_hg / HG_PER_TONNE, y_pred_hg / HG_PER_TONNE
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9BAA9F")
    ax.tick_params(colors=INK)


def plot_training_loss(history, path):
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    epochs = np.arange(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], color=GREEN, linewidth=2, label="Training loss")
    ax.plot(epochs, history["val_loss"], color=AMBER, linewidth=2, label="Validation loss")
    best = int(np.argmin(history["val_loss"]))
    ax.scatter([best + 1], [history["val_loss"][best]], color=INK, zorder=5, s=28,
               label=f"Best epoch ({best + 1})")
    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel("Loss (MSE on standardised yield)", color=INK)
    ax.set_title("Training vs validation loss", color=INK, loc="left", fontweight="bold")
    ax.set_yscale("log")  # log scale so the late, small improvements stay visible
    ax.yaxis.set_major_locator(FixedLocator([0.015, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    _style(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_actual_vs_predicted(y_true_t, y_pred_t, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=140)
    ax.scatter(y_true_t, y_pred_t, s=7, alpha=0.35, color=GREEN, edgecolors="none")
    lim = [0, max(y_true_t.max(), y_pred_t.max()) * 1.03]
    ax.plot(lim, lim, color=AMBER, linewidth=1.6, label="Perfect prediction")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual yield (t/ha)", color=INK)
    ax.set_ylabel("Predicted yield (t/ha)", color=INK)
    ax.set_title("Actual vs predicted yield (test set)", color=INK, loc="left", fontweight="bold")
    _style(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_residuals(y_true_t, y_pred_t, path):
    errors = y_pred_t - y_true_t
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=140)
    ax.hist(errors, bins=60, color=GREEN, alpha=0.85, edgecolor="white")
    ax.axvline(0, color=AMBER, linewidth=1.8)
    ax.set_xlabel("Prediction error = predicted - actual (t/ha)", color=INK)
    ax.set_ylabel("Number of test samples", color=INK)
    ax.set_title("Error distribution (test set)", color=INK, loc="left", fontweight="bold")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def pick_examples(X_test, y_test, predictions_t):
    """Real rows from the TEST set: two India/Potatoes rows (different years) + other crops."""
    frame = X_test.copy()
    frame["actual_t"] = y_test.values / HG_PER_TONNE
    frame["predicted_t"] = predictions_t
    india = frame[(frame.country == "India") & (frame.crop == "Potatoes")].sort_values("year")
    # earliest and latest year available in the test set (so the two rows differ)
    chosen = india.iloc[[0, -1]] if india.year.nunique() > 1 else india.tail(1)
    rest = frame.drop(chosen.index).sample(frac=1, random_state=42)
    seen_crops = set(chosen.crop)
    picks = [chosen]
    for _, row in rest.iterrows():
        if row.crop not in seen_crops and len(seen_crops) < 6:
            seen_crops.add(row.crop)
            picks.append(row.to_frame().T)
    examples = pd.concat(picks)
    return [
        {
            "country": r.country,
            "crop": r.crop,
            "year": int(r.year),
            "rainfall": float(r.rainfall),
            "pesticides": float(r.pesticides),
            "temperature": float(r.temperature),
            "actual_t_ha": round(float(r.actual_t), 2),
            "predicted_t_ha": round(float(r.predicted_t), 2),
        }
        for r in examples.itertuples()
    ]


def run_evaluation(model=None, preprocessor=None, meta=None):
    from tensorflow import keras  # imported here so plain helper imports stay light

    RESULTS_DIR.mkdir(exist_ok=True)
    if meta is None:
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if model is None:
        model = keras.models.load_model(MODEL_PATH)
    if preprocessor is None:
        preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Same cleaning + same seed  ->  exactly the same split as in train.py
    df, _ = load_and_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    pred_test = predict_hg_ha(model, preprocessor, meta, X_test)
    pred_train = predict_hg_ha(model, preprocessor, meta, X_train)
    test = score(y_test.values, pred_test)
    train = score(y_train.values, pred_train)

    y_true_t, y_pred_t = y_test.values / HG_PER_TONNE, pred_test / HG_PER_TONNE
    plot_training_loss(meta["history"], RESULTS_DIR / "training_loss.png")
    plot_actual_vs_predicted(y_true_t, y_pred_t, RESULTS_DIR / "actual_vs_predicted.png")
    plot_residuals(y_true_t, y_pred_t, RESULTS_DIR / "residuals.png")

    examples = pick_examples(X_test, y_test, y_pred_t)

    # ---- text report --------------------------------------------------------
    hist = meta["history"]
    best = int(np.argmin(hist["val_loss"]))
    lines = [
        "Deep Learning Based Crop Yield Prediction System - model metrics",
        "=" * 64,
        f"Test samples     : {len(X_test)}   (never used for training or tuning)",
        f"Training samples : {len(X_train)}",
        "",
        "Metrics in tonnes/hectare (t/ha)          TEST      TRAIN",
        f"  MAE  (mean absolute error)         {test['mae']:8.4f}   {train['mae']:8.4f}",
        f"  MSE  (mean squared error)          {test['mse']:8.4f}   {train['mse']:8.4f}",
        f"  RMSE (root mean squared error)     {test['rmse']:8.4f}   {train['rmse']:8.4f}",
        f"  R2   (coefficient of determination){test['r2']:8.4f}   {train['r2']:8.4f}",
        "",
        f"MAE in the dataset's own unit: {test['mae'] * HG_PER_TONNE:,.0f} hg/ha  (1 t/ha = 10,000 hg/ha)",
        "",
        "Loss values are Mean Squared Error on the STANDARDISED yield",
        "(yield is standardised for training, then converted back for the metrics above).",
        f"  Epochs run            : {len(hist['loss'])}",
        f"  Best epoch            : {best + 1}",
        f"  Training loss (best)  : {hist['loss'][best]:.5f}",
        f"  Validation loss (best): {hist['val_loss'][best]:.5f}",
    ]
    (RESULTS_DIR / "model_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sample = [
        "Test cases - real rows from the TEST set (unseen during training)",
        "=" * 96,
        f"{'Country':<16}{'Crop':<16}{'Year':>5}{'Rain(mm)':>10}{'Pest(t)':>11}{'Temp':>7}{'Actual':>9}{'Predicted':>11}{'Error':>8}",
    ]
    for e in examples:
        sample.append(
            f"{e['country']:<16}{e['crop']:<16}{e['year']:>5}{e['rainfall']:>10.0f}{e['pesticides']:>11.0f}"
            f"{e['temperature']:>7.2f}{e['actual_t_ha']:>9.2f}{e['predicted_t_ha']:>11.2f}"
            f"{e['predicted_t_ha'] - e['actual_t_ha']:>+8.2f}"
        )
    sample += [
        "",
        "Yields are in tonnes/hectare. 'Actual' is the value recorded in the dataset.",
        "The model output is an ESTIMATE; compare it with known dataset values like these,",
        "not with a random number from the internet (different sources use different data).",
    ]
    (RESULTS_DIR / "sample_predictions.txt").write_text("\n".join(sample) + "\n", encoding="utf-8")

    # ---- save into metadata so the web app's performance page can show it --------
    meta["metrics"] = {"test": test, "train": train}
    meta["examples"] = examples
    meta["split_sizes"] = {"train": len(X_train), "validation": len(X_val), "test": len(X_test)}
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print()
    print("\n".join(sample))
    return meta


if __name__ == "__main__":
    run_evaluation()
