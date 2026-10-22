"""Train and evaluate the RetainIQ churn classification prototype."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "churn_dataset.csv"
MODEL_PATH = BASE_DIR / "ml" / "churn_model.joblib"

TARGET = "churn_within_30_days"

CATEGORICAL_FEATURES = ["plan"]
NUMERIC_FEATURES = [
    "monthly_price",
    "tenure_months",
    "logins_last_30_days",
    "feature_uses_last_30_days",
    "support_requests_last_30_days",
    "days_since_last_activity",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run `python ml/generate_dataset.py` first."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    if df[TARGET].nunique() != 2:
        raise ValueError("Expected exactly two target classes: 0 and 1.")

    missing_features = set(FEATURES) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {sorted(missing_features)}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "plan",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("===== RETAINIQ CHURN MODEL =====")
    print(f"Dataset rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Positive churn rate: {y.mean():.1%}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            zero_division=0,
        )
    )

    print("===== CONFUSION MATRIX =====")
    print("Rows = actual class; columns = predicted class")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\n===== ROC-AUC =====")
    print(f"{roc_auc_score(y_test, probabilities):.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
