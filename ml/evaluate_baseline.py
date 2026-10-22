"""Compare RetainIQ's trained model with a feature-agnostic baseline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "churn_dataset.csv"
MODEL_PATH = BASE_DIR / "ml" / "churn_model.joblib"

TARGET = "churn_within_30_days"
FEATURES = [
    "plan",
    "monthly_price",
    "tenure_months",
    "logins_last_30_days",
    "feature_uses_last_30_days",
    "support_requests_last_30_days",
    "days_since_last_activity",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    trained_model = joblib.load(MODEL_PATH)

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)

    models = [
        ("Logistic regression", trained_model),
        ("Dummy baseline", baseline),
    ]

    for name, model in models:
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        print(f"\n===== {name} =====")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["No churn", "Churn"],
                zero_division=0,
            )
        )


if __name__ == "__main__":
    main()
