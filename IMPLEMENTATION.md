# Implementation Guide

## Overview

CoverSignal is a local insurance purchase-prediction demonstration. A seeded Python generator creates the training CSV, scikit-learn trains a preprocessing and Logistic Regression pipeline, FastAPI serves the saved pipeline and static web page, and browser JavaScript submits validated profiles to the prediction endpoint.

```text
generate_data.py
      |
      v
data/insurance_purchases.csv --> train_model.py --> model.pkl
                                                       |
Browser: static/index.html + app.js --> FastAPI --> prediction
                                  styles.css        /api/predict
```

## Project Files

- `generate_data.py`: creates the reproducible sample dataset.
- `data/insurance_purchases.csv`: generated 5,000-row training dataset.
- `train_model.py`: loads and preprocesses the data, evaluates the model, and writes `model.pkl`.
- `model.pkl`: serialized scikit-learn pipeline, including preprocessing and classifier.
- `main.py`: FastAPI application, request validation, static file serving, and prediction endpoint.
- `static/index.html`: profile form and prediction result structure.
- `static/styles.css`: responsive page layout and visual styling.
- `static/app.js`: browser validation, API request, and result updates.
- `requirements.txt`: runtime and training dependencies.

## Sample Data

Run `python generate_data.py` to create or replace `data/insurance_purchases.csv`. The generator uses Python's seeded random number generator (`20261013`) and writes 5,000 rows. Its columns are:

| Column                 | Type     | Generated values                                                                      |
| ---------------------- | -------- | ------------------------------------------------------------------------------------- |
| `age`                | integer  | 18-79 years                                                                           |
| `annual_income`      | number   | Normal distribution centered at $68,000, bounded to $18,000-$180,000, rounded to $500 |
| `annual_premium`     | number   | Normal distribution centered at $950, bounded to $250-$2,500, rounded to $10          |
| `vehicle_age_years`  | integer  | Exponential distribution, rounded and capped at 22                                    |
| `previously_insured` | 0 or 1   | Bernoulli draw with 42% probability of 1                                              |
| `vehicle_damage`     | 0 or 1   | Probability increases with vehicle age, capped at 62%                                 |
| `region`             | category | Central, Coastal, Mountain, or Metro, selected uniformly                              |
| `purchased`          | 0 or 1   | Sampled from the synthetic purchase probability                                       |

The purchase probability is generated from a logistic function. The log-odds increase with age, income, vehicle age, vehicle damage, and region-specific effects; they decrease with premium and prior insurance. Random sampling around that probability creates a binary target rather than a perfectly deterministic label.

These relationships are invented for demonstrating the software path. They are not estimated from real customers and should not be interpreted as actuarial assumptions.

## Model Training

Run `python train_model.py`. If the CSV is absent, training invokes the data generator. It then validates that the expected feature and target columns exist and that both target classes are present.

The data is split into 80% training and 20% holdout data with stratification and a fixed split seed. The scikit-learn pipeline applies:

1. `StandardScaler` to age, income, premium, and vehicle age.
2. Passthrough encoding to the two binary history fields.
3. `OneHotEncoder(handle_unknown="ignore")` to region.
4. `LogisticRegression(max_iter=1000, random_state=42)` as the classifier.

Keeping preprocessing inside the pipeline ensures training and prediction use the same transformations. The holdout accuracy and ROC AUC are printed for a basic demonstration check. The complete fitted pipeline is serialized to `model.pkl` with joblib.

## Backend Contract

Start the service with `uvicorn main:app --reload`. FastAPI loads `model.pkl` when the module starts. If it is missing, health reports `model_loaded: false` and predictions return HTTP 503 until the model is trained and the app restarted.

### Endpoints

| Method and path       | Behavior                                                           |
| --------------------- | ------------------------------------------------------------------ |
| `GET /`             | Serves the HTML application.                                       |
| `GET /api/health`   | Returns service status and whether the model loaded.               |
| `POST /api/predict` | Validates a profile and returns a predicted class and probability. |
| `GET /docs`         | Serves FastAPI's interactive API documentation.                    |

Example prediction request:

```json
{
  "age": 38,
  "annual_income": 68000,
  "annual_premium": 950,
  "vehicle_age_years": 6,
  "previously_insured": false,
  "vehicle_damage": false,
  "region": "Central"
}
```

Successful responses contain `prediction` (`0` or `1`), `probability` (purchase probability from 0 to 1), and `result` (`"Likely to purchase"` or `"Unlikely to purchase"`). The class uses a 0.5 probability threshold. Pydantic enforces the same numeric ranges as the form, limits region to the four supported categories, and rejects unexpected fields. Invalid requests return HTTP 422.

## Frontend Flow

The page is served by FastAPI, so the form and API share one origin. Native HTML constraints provide immediate range and required-field feedback. On submission, `app.js` builds the JSON profile, disables the button while the request is pending, and posts to `/api/predict`. On success it updates the percentage, classification, and progress indicator; on failure it displays the returned error message.

## Run the Application

From the project root on macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_data.py
python train_model.py
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The API health check is [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health).

## Limitations and Safety

The dataset and model are instructional samples. Their metrics describe only a random holdout from the same synthetic process and do not establish real-world predictive performance, calibration, fairness, or business value. Do not use the output for underwriting, premium setting, eligibility, or decisions about real people. `model.pkl` uses Python pickle serialization through joblib; only load model files from trusted sources, and retrain with compatible dependency versions when changing environments.
