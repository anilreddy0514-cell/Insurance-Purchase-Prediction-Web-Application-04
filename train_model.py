"""Train and save the insurance purchase Logistic Regression pipeline."""

import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generate_data import DATA_PATH, generate_data


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
TARGET = "purchased"
NUMERIC_FEATURES = ["age", "annual_income", "annual_premium", "vehicle_age_years"]
BINARY_FEATURES = ["previously_insured", "vehicle_damage"]
CATEGORICAL_FEATURES = ["region"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def load_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    if not DATA_PATH.exists():
        generate_data()

    with DATA_PATH.open(newline="", encoding="utf-8") as dataset_file:
        records = list(csv.DictReader(dataset_file))

    if not records or any(feature not in records[0] for feature in FEATURES + [TARGET]):
        raise ValueError(f"Dataset must include these columns: {', '.join(FEATURES + [TARGET])}")

    features: list[dict[str, object]] = []
    labels: list[int] = []
    for record in records:
        features.append(
            {
                **{name: float(record[name]) for name in NUMERIC_FEATURES},
                **{name: int(record[name]) for name in BINARY_FEATURES},
                "region": record["region"],
            }
        )
        labels.append(int(record[TARGET]))

    return pd.DataFrame(features), np.asarray(labels)


def train_model() -> None:
    features, labels = load_dataset()
    if len(np.unique(labels)) != 2:
        raise ValueError("Training data must contain both purchased classes (0 and 1).")

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    joblib.dump(model, MODEL_PATH)

    print(f"Rows: {len(labels):,} | purchase rate: {labels.mean():.1%}")
    print(f"Holdout accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Holdout ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()