# Deep Learning Based Crop Yield Prediction System

**Predicting Agricultural Crop Yield Using Environmental and Agricultural Factors**

A simple, explainable Deep Learning project (Advanced Deep Learning course). A Dense neural network (MLP) built with TensorFlow/Keras predicts crop yield in **tonnes per hectare (t/ha)**. A Flask web app serves the trained model through a modern dashboard.

---

## 1. Project Overview

Crop yield depends on many factors: the country, the crop, the weather and farming inputs. This project trains a neural network on historical records so it can **estimate** the yield for a new combination of inputs.

- The model is trained **once** (`python train.py`).
- The web app only **loads** the saved model. It never retrains.
- Everything runs locally. No internet, API key, database or Docker is needed while the app runs.

## 2. Objective

Predict crop yield (t/ha) from: **Country, Crop, Year, Average rainfall, Pesticide usage, Average temperature**, using a Deep Learning regression model.

## 3. Dataset

| Item | Detail |
|---|---|
| Name | Crop Yield Prediction Dataset (Kaggle, built from FAO and World Bank data) |
| Source page | https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset |
| File used | `yield_df.csv`, included here as `dataset/crop_yield.csv` |
| Also downloadable from | https://raw.githubusercontent.com/Aishwarya-3092/yeild-prediction/main/yield_df.csv (a public copy of the same file) |
| Size | 28,242 raw rows, 25,932 after removing 2,310 duplicate rows |
| Coverage | 101 countries, 10 crops, years 1990 to 2013 |

> **Licence note:** the Kaggle page is the authoritative source for the dataset's licence. Check it before publishing this project publicly (for example on GitHub). For a college submission, cite the source page above.

**Original columns**

| Column in CSV | Meaning | Used as |
|---|---|---|
| `Unnamed: 0` | Row index | dropped |
| `Area` | Country | input (categorical) |
| `Item` | Crop | input (categorical) |
| `Year` | Year | input (numeric) |
| `average_rain_fall_mm_per_year` | Rainfall in mm per year | input (numeric) |
| `pesticides_tonnes` | Pesticide use in tonnes | input (numeric) |
| `avg_temp` | Average temperature in °C | input (numeric) |
| `hg/ha_yield` | Yield in **hectograms per hectare** | **target** |

## 4. Features (inputs)

- **Country:** where the crop was grown (101 options).
- **Crop:** Cassava, Maize, Plantains and others, Potatoes, Rice (paddy), Sorghum, Soybeans, Sweet potatoes, Wheat, Yams.
- **Year:** harvest year (1990 to 2013).
- **Rainfall:** average annual rainfall in mm/year.
- **Pesticides:** pesticide use in tonnes. In this dataset it is a country-level total.
- **Temperature:** average temperature in °C.

## 5. Target and units

- The dataset stores yield in **hg/ha** (hectograms per hectare).
- **1 tonne = 10,000 hectograms**, so `t/ha = hg/ha ÷ 10,000`.
- The network is trained on a standardised yield (subtract the training mean, divide by the training standard deviation) because that trains more stably. `app.py` reverses this exactly: network output → hg/ha → t/ha.
- Nothing is multiplied or scaled to make results "look realistic". The only extra rule is that a predicted yield below zero is shown as zero, since a negative yield is impossible.

## 6. Deep Learning architecture

```text
Country ────────┐
Crop ───────────┤
Year ───────────┤
Rainfall ───────┼──►  Preprocessing  ──►  115 numbers
Pesticides ─────┤     (one-hot + scaling)
Temperature ────┘
                              │
                              ▼
                   Dense 128  (ReLU)
                              ▼
                   Dropout 0.20
                              ▼
                   Dense 64   (ReLU)
                              ▼
                   Dropout 0.20
                              ▼
                   Dense 32   (ReLU)
                              ▼
                   Dense 1    (Linear)
                              ▼
                   Crop Yield (t/ha)
```

A picture version is in `static/architecture.svg` (also shown on the Model Performance page).

| Layer | Purpose |
|---|---|
| Input (115 features) | 101 country columns + 10 crop columns (one-hot) + 4 scaled numbers |
| Dense 128, ReLU | First layer that learns combinations of inputs |
| Dropout 0.20 | Randomly switches off 20% of neurons during training to reduce overfitting |
| Dense 64, ReLU | Learns higher-level patterns |
| Dropout 0.20 | Second overfitting guard |
| Dense 32, ReLU | Compresses the patterns |
| Dense 1, Linear | One number: the predicted (standardised) yield |

- Total trainable parameters: **25,217**
- Optimizer: **Adam** (learning rate 0.001)
- Loss: **Mean Squared Error**; metric: **Mean Absolute Error**
- Batch size: 64, maximum 300 epochs, with **EarlyStopping** (patience 25, restores the best weights) and **ModelCheckpoint** (saves the best model)
- Random seed: 42

## 7. Preprocessing (no data leakage)

Order used in `train.py`:

```text
Dataset → clean → Train / Validation / Test split → fit preprocessing on TRAIN only
        → transform train, validation, test → train network → evaluate on TEST
```

- **Cleaning:** drop the index column, fix data types, drop missing values (none found), drop 2,310 duplicate rows, remove physically impossible values (none found). Very high yields are real (for example potatoes and cassava), so they are **not** cut as outliers.
- **Country, Crop:** `OneHotEncoder` (unknown values are ignored safely). No hand-written encodings.
- **Year, Rainfall, Temperature:** `StandardScaler`.
- **Pesticides:** `log(1 + x)` then `StandardScaler`, because values range from 0.04 to 367,778 tonnes and a few huge values would otherwise dominate.
- **Split:** 20% test; the rest split 85% train / 15% validation, giving 17,633 train, 3,112 validation, 5,187 test rows.

## 8. Technologies

Python, TensorFlow/Keras, Flask, HTML, CSS, JavaScript, pandas, NumPy, scikit-learn, Matplotlib, joblib.

## 9. Versions (tested set)

| Package | Version |
|---|---|
| Python | 3.10, 3.11 or 3.12 (built and tested on 3.12.3) |
| TensorFlow | 2.18.1 |
| Keras | 3.8.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.5.2 |
| Flask | 3.0.3 |
| Matplotlib | 3.9.2 |
| joblib | 1.4.2 |

These versions are compatible with each other. The neural network is saved as `.keras`. The preprocessor is saved with joblib as `preprocessor.pkl`; `models/metadata.json` records the versions it was created with, and the app prints a warning if the installed scikit-learn differs.

> **The "incompatible dtype" pickle error:** it appears when a scikit-learn object saved with one version is loaded with another. Avoid it by installing the exact versions in `requirements.txt` and running `python train.py` on **your own machine**. Then the model and preprocessor are created and loaded with the same versions.

## 10. Installation (Windows 11, PowerShell)

Use Python 3.10, 3.11 or 3.12 (64-bit). Check with `python --version`.

```powershell
cd crop-yield-prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 11. Training

```powershell
python train.py
```

This takes a few minutes on a normal CPU. It prints each step, saves everything in `models/` and `results/`, and runs the evaluation automatically.

## 12. Running the application

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

Try: click one of the **"Load a real record"** chips (for example *India, Potatoes, 2013*) and press **Predict Crop Yield**. The app shows the estimate, the exact inputs used, and the value recorded in the dataset for that held-out record.

Other pages and endpoints:

| Route | Purpose |
|---|---|
| `GET /` | Prediction dashboard |
| `GET /performance` | Model Performance page (metrics, graphs, architecture, test cases) |
| `POST /predict` | JSON API |
| `GET /api/options` | Dropdown values and data ranges |
| `GET /health` | Is the model loaded? |

**API example**

```json
POST /predict
{
  "country": "India", "crop": "Potatoes",
  "rainfall": 1083, "pesticides": 45620, "temperature": 26.71, "year": 2013
}
```

Response (shortened):

```json
{
  "predicted_yield_tonnes_per_hectare": 22.1,
  "predicted_yield_kg_per_hectare": 22102,
  "inputs": { "country": "India", "crop": "Potatoes", "...": "..." },
  "warnings": [],
  "note": "AI/Deep Learning estimate based on historical training data, not a guaranteed outcome."
}
```

**Input checks:** empty fields, non-numbers, negative rainfall or pesticides, temperature outside -50 to 60 °C, invalid years, and unknown countries or crops all return a friendly message. Values that are allowed but outside the training range (or a country/crop pair never seen in training) return a **warning** and the estimate is less reliable.

## 13. Model evaluation (test set, never used for training)

| Metric | Test | Train |
|---|---|---|
| MAE | **0.67 t/ha** | 0.59 t/ha |
| MSE | 1.53 | 1.05 |
| RMSE | **1.24 t/ha** | 1.03 t/ha |
| R² | **0.979** | 0.985 |

- Stopped after 60 epochs; best epoch 35 (validation loss 0.0178 on the standardised target).
- Your own numbers will be very close but may differ slightly after retraining.

Files written by training and evaluation (`results/`):

| File | What it shows |
|---|---|
| `training_loss.png` | Training vs validation loss per epoch. Both fall and level out, so the network is not just memorising. |
| `actual_vs_predicted.png` | Each test sample: actual vs predicted yield. Points near the diagonal are accurate. |
| `residuals.png` | Distribution of errors (predicted minus actual). Centred near zero. |
| `model_metrics.txt` | The metric table above |
| `sample_predictions.txt` | Test cases from real dataset rows |

## 14. Test cases (real rows from the test set)

These rows were **not** used for training. Yields are in t/ha. Run `python evaluate.py` to regenerate them.

| Country | Crop | Year | Rainfall (mm) | Pesticides (t) | Temp (°C) | Actual | Predicted |
|---|---|---|---|---|---|---|---|
| India | Potatoes | 1990 | 1,083 | 75,000 | 25.16 | 15.71 | 15.97 |
| India | Potatoes | 2013 | 1,083 | 45,620 | 26.71 | 22.76 | 22.10 |
| India | Sorghum | 1990 | 1,083 | 75,000 | 26.36 | 0.81 | 1.00 |
| Colombia | Soybeans | 1994 | 3,240 | 17,051 | 27.17 | 1.93 | 1.88 |
| Brazil | Cassava | 2010 | 1,761 | 342,580 | 26.30 | 13.95 | 14.32 |
| Cameroon | Rice, paddy | 2010 | 1,604 | 1,161 | 25.70 | 1.09 | 2.46 |
| Turkey | Maize | 1998 | 593 | 30,162 | 14.48 | 4.22 | 4.21 |

Predictions are **estimates**. Judge the model by comparing it with known dataset values like these (or by its test metrics), not by comparing it with a random number from the internet, because other sources use different data and definitions. The Cameroon row shows that individual predictions can miss by more than the average error.

## 15. Screenshots

Add your own screenshots here after running the app:

- `![Prediction dashboard](screenshots/dashboard.png)`
- `![Prediction result](screenshots/result.png)`
- `![Model performance](screenshots/performance.png)`

## 16. Project structure

```text
crop-yield-prediction/
├── app.py                 Flask app (loads the trained model)
├── train.py               trains the network and saves everything
├── evaluate.py            test-set metrics, graphs, sample predictions
├── data_utils.py          shared cleaning, splitting and preprocessing
├── requirements.txt
├── README.md
├── dataset/crop_yield.csv
├── models/                crop_yield_model.keras, preprocessor.pkl, metadata.json
├── notebooks/crop_yield_training.ipynb
├── templates/             index.html, performance.html
├── static/                style.css, script.js, architecture.svg
└── results/               graphs, model_metrics.txt, sample_predictions.txt
```

`data_utils.py` is the one file added to your suggested structure. It keeps the cleaning and splitting code in one place so `train.py` and `evaluate.py` cannot disagree.

## 17. Limitations

- **Dataset:** 1990 to 2013 only, 101 countries and 10 crops. It cannot know about newer years, varieties or technology.
- **Country-level data:** rainfall, temperature and pesticides are country-level values, not farm-level. The model cannot capture soil, irrigation, seed variety, fertiliser or local weather.
- **Optimistic score:** the split is random, and the dataset has near-duplicate rows (the same country and year appear for several crops with the same weather values, and some country/crop/year rows repeat with slightly different temperatures). Similar rows can land in both train and test, so R² 0.979 is probably higher than what you would get on truly new years or countries.
- **Historical data:** predictions describe patterns in past records, not the future.
- **Out-of-range inputs:** values outside the training range or unseen country/crop pairs give unreliable results (the app warns you).
- **Estimates only:** results are not guaranteed and are **not professional agricultural advice**.

---

## 18. How to Explain This Project to a Professor

**Problem.** Crop yield depends on many agricultural and environmental variables, so estimating it by hand is difficult.

**Solution.** A neural network learns the relationship between the inputs (country, crop, year, rainfall, pesticides, temperature) and historical yield from about 26,000 records.

**Why Deep Learning?** A neural network can learn non-linear relationships and interactions between several numerical and categorical features. For example, the effect of rainfall is different for rice than for wheat.

**Why regression?** The output is a continuous number such as 3.25 t/ha, not a class such as Rice, Wheat or Potato.

**Why ReLU?** ReLU is `max(0, x)`. It is simple and fast, lets the network learn non-linear patterns, and avoids the vanishing-gradient problem that sigmoid/tanh have in deeper networks.

**Why a linear output?** A regression output must be able to take any real value. A sigmoid would squash it into 0 to 1 and ReLU could never go below zero; linear leaves the value unrestricted.

**Why MSE?** It measures the average squared error, is smooth and differentiable (good for gradient descent), and punishes large mistakes more than small ones.

**The flow to describe.** Clean data → split → fit preprocessing on training data → train the MLP with early stopping → evaluate on unseen test data → save the model → the Flask app loads it and converts the output from hg/ha to t/ha.

## 19. Viva questions and answers

1. **What is crop yield?** The amount of crop produced per unit of land, here tonnes per hectare.
2. **Why is this a regression problem?** The output is a continuous number, not a category.
3. **Why did you use a neural network?** It can learn non-linear relationships between many mixed inputs, and it satisfies the Deep Learning requirement of the course.
4. **Why ReLU?** Simple, fast, non-linear, and it reduces vanishing gradients.
5. **Why is the final layer linear?** Regression needs an unbounded real-valued output.
6. **What is an epoch?** One full pass of the training data through the network.
7. **What is batch size?** The number of samples processed before the weights are updated. Here it is 64.
8. **What is overfitting?** The model memorises the training data and does poorly on new data.
9. **How did you prevent overfitting?** Dropout (0.2), EarlyStopping on validation loss, a separate validation set, and a held-out test set.
10. **Why dropout?** It randomly disables neurons during training, so the network cannot depend on any single neuron.
11. **What is MAE?** Mean Absolute Error: the average size of the error. Here about 0.67 t/ha.
12. **What is RMSE?** The square root of MSE. It is in the same unit as the target and penalises large errors more. Here about 1.24 t/ha.
13. **What is R²?** The fraction of the variation in yield that the model explains. 1 is perfect, 0 is no better than predicting the average. Here 0.979.
14. **Why normalise numerical features?** Features with very different scales (pesticides in the thousands, temperature in tens) would otherwise dominate training and slow it down.
15. **How are categorical variables handled?** One-hot encoding for country and crop, done by `OneHotEncoder`, not by hand.
16. **Training vs testing data?** The network learns from training data; the test data is kept unseen to measure real performance. A validation set is used during training to decide when to stop.
17. **What is data leakage?** Information from test data influencing training, for example fitting the scaler on all data before splitting. This project fits preprocessing on training data only.
18. **Why tonnes/hectare?** It is the standard unit for yield and easy to read. The dataset uses hg/ha, so the app divides by 10,000.
19. **What are the limitations?** Country-level data, years 1990 to 2013, an optimistic random split, and estimates that are not agricultural advice.
20. **What happens for values outside the training range?** Neural networks extrapolate poorly, so the estimate is unreliable. The app still answers but shows a warning.
21. **Why standardise the target?** It makes training more stable. The app converts the output back to hg/ha and then t/ha.
22. **Why log-transform pesticides?** Its values are extremely skewed (0.04 to 367,778); the log stops a few huge values dominating.
23. **Why is the model not retrained when Flask starts?** Training is slow; the app just loads the saved `.keras` model and `preprocessor.pkl`.
24. **Why not Random Forest or XGBoost?** The course requires a Deep Learning model, and a single MLP is easier to explain and debug than an ensemble.

## 20. Troubleshooting

- **"The trained model was not found":** run `python train.py` first.
- **Model fails to load or a dtype error appears:** you are using different library versions. Reinstall from `requirements.txt` and run `python train.py` again.
- **`pip install tensorflow` fails:** make sure Python is 64-bit and version 3.10 to 3.12.
- **Port 5000 already in use:** close the other program, or change the port in the last line of `app.py`.
- **Graphs missing on the Model Performance page:** run `python evaluate.py`.

## 21. Credits

Dataset: Kaggle *Crop Yield Prediction Dataset* (FAO and World Bank data). Model, application and documentation: this project.
