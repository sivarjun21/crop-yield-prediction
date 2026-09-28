"""
data_utils.py
-------------
Shared helpers used by train.py and evaluate.py:
  * load_and_clean()      -> read the CSV and clean it
  * split_data()          -> train / validation / test split (fixed seed)
  * build_preprocessor()  -> encoders + scalers (fitted ONLY on training data)

The Flask app (app.py) does not import this file. It only loads the saved
preprocessor.pkl and the trained .keras model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "crop_yield.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_PATH = MODELS_DIR / "crop_yield_model.keras"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

RANDOM_STATE = 42          # the same seed everywhere -> reproducible results
HG_PER_TONNE = 10_000      # 1 tonne = 10,000 hectograms  (so 1 hg/ha = 0.0001 t/ha)

# Friendly column names used inside the project
RENAME_COLUMNS = {
    "Area": "country",
    "Item": "crop",
    "Year": "year",
    "average_rain_fall_mm_per_year": "rainfall",
    "pesticides_tonnes": "pesticides",
    "avg_temp": "temperature",
    "hg/ha_yield": "yield_hg_ha",
}
CATEGORICAL = ["country", "crop"]
NUMERIC = ["year", "rainfall", "pesticides", "temperature"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "yield_hg_ha"     # the dataset stores yield in hectograms per hectare


# ----------------------------------------------------------------------------
# 1. Load + clean
# ----------------------------------------------------------------------------
def load_and_clean(path=DATASET_PATH):
    """Read the CSV and return (clean_dataframe, cleaning_report)."""
    df = pd.read_csv(path)
    report = {"raw_rows": int(len(df))}

    # The Kaggle file has an unnamed index column ("Unnamed: 0") -> drop it.
    # This must happen BEFORE duplicate removal, otherwise no row looks duplicated.
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df = df.rename(columns=RENAME_COLUMNS)

    # Make sure every column has the right data type.
    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()

    # Missing values (unparseable numbers become NaN above)
    report["rows_with_missing_values"] = int(df.isna().any(axis=1).sum())
    df = df.dropna()

    # Duplicate rows
    report["duplicate_rows_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Basic outlier / sanity handling: remove physically impossible values only.
    # Very high yields (e.g. potatoes, cassava) are real, so we do NOT cut them.
    valid = (
        (df[TARGET] > 0)
        & (df["rainfall"] > 0)
        & (df["pesticides"] >= 0)
        & df["temperature"].between(-50, 60)
    )
    report["impossible_rows_removed"] = int((~valid).sum())
    df = df[valid].copy()

    df["year"] = df["year"].astype(int)
    report["clean_rows"] = int(len(df))
    return df.reset_index(drop=True), report


# ----------------------------------------------------------------------------
# 2. Split  (BEFORE any fitting -> no data leakage)
# ----------------------------------------------------------------------------
def split_data(df):
    """
    80% of rows are set aside for training + validation, 20% for the final test.
    The 80% part is split again: 85% train / 15% validation
    (about 68% train, 12% validation, 20% test overall).
    """
    X = df[FEATURES]
    y = df[TARGET]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------------------------------------------------------
# 3. Preprocessing pipeline
# ----------------------------------------------------------------------------
def build_preprocessor():
    """
    country, crop            -> One-Hot Encoding (unknown values are ignored safely)
    year, rainfall, temp     -> StandardScaler   (mean 0, standard deviation 1)
    pesticides               -> log(1 + x) then StandardScaler
                                (pesticide use ranges from 0.04 to 367,778 tonnes,
                                 so the log keeps a few huge values from dominating)
    """
    return ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("scaled_numbers", StandardScaler(), ["year", "rainfall", "temperature"]),
            (
                "log_pesticides",
                Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]),
                ["pesticides"],
            ),
        ]
    )
