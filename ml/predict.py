"""Inference helpers for the RetainIQ churn prototype."""

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "churn_model.joblib"

FEATURE_COLUMNS = [
    "plan",
    "monthly_price",
    "tenure_months",
    "logins_last_30_days",
    "feature_uses_last_30_days",
    "support_requests_last_30_days",
    "days_since_last_activity",
]

MODEL_NOTICE = (
    "Prototype trained on synthetic data. "
    "Predictions are not validated for real-world customer churn."
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `python ml/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_churn(model, features: dict) -> dict:
    row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    probability = float(model.predict_proba(row)[0, 1])
    predicted_churn = bool(model.predict(row)[0])

    if probability >= 0.70:
        risk_level = "high"
    elif probability >= 0.40:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "churn_probability": probability,
        "risk_level": risk_level,
        "predicted_churn": predicted_churn,
        "model_notice": MODEL_NOTICE,
    }
