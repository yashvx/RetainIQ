"""Generate synthetic customer data for RetainIQ ML development.

All churn outcomes are simulated for pipeline development, not real-world
prediction or production performance claims.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_CUSTOMERS = 1500
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "churn_dataset.csv"


def generate_dataset(
    n_customers: int = N_CUSTOMERS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    plans = rng.choice(
        ["basic", "pro", "enterprise"],
        size=n_customers,
        p=[0.50, 0.35, 0.15],
    )

    plan_prices = {"basic": 499.0, "pro": 999.0, "enterprise": 1999.0}
    monthly_price = np.round(
        np.array([plan_prices[plan] for plan in plans])
        * rng.uniform(0.85, 1.15, size=n_customers),
        2,
    )

    tenure_months = rng.integers(1, 49, size=n_customers)
    logins_last_30_days = np.clip(
        rng.poisson(lam=12, size=n_customers), 0, 60
    )
    feature_uses_last_30_days = np.clip(
        rng.poisson(lam=20, size=n_customers), 0, 100
    )
    support_requests_last_30_days = np.clip(
        rng.poisson(lam=1.2, size=n_customers), 0, 10
    )
    days_since_last_activity = np.clip(
        rng.exponential(scale=7, size=n_customers).round().astype(int),
        0,
        60,
    )

    # A transparent simulated relationship:
    # more inactivity/support requests increase risk;
    # more logins, feature use, and tenure reduce risk.
    log_odds = (
        -1.3
        + 0.09 * (days_since_last_activity - 7)
        - 0.25 * (logins_last_30_days - 12)
        - 0.08 * (feature_uses_last_30_days - 20)
        + 0.35 * (support_requests_last_30_days - 1.2)
        - 0.015 * (tenure_months - 24)
    )

    churn_probability = 1 / (1 + np.exp(-log_odds))
    churn_within_30_days = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "plan": plans,
            "monthly_price": monthly_price,
            "tenure_months": tenure_months,
            "logins_last_30_days": logins_last_30_days,
            "feature_uses_last_30_days": feature_uses_last_30_days,
            "support_requests_last_30_days": support_requests_last_30_days,
            "days_since_last_activity": days_since_last_activity,
            "churn_within_30_days": churn_within_30_days,
        }
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print("\nChurn label distribution:")
    print(dataset["churn_within_30_days"].value_counts().sort_index())
    print(f"\nChurn rate: {dataset['churn_within_30_days'].mean():.1%}")


if __name__ == "__main__":
    main()
