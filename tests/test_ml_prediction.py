from ml.predict import load_model, predict_churn


def test_prediction_returns_expected_fields():
    model = load_model()

    features = {
        "plan": "basic",
        "monthly_price": 29.99,
        "tenure_months": 4,
        "logins_last_30_days": 2,
        "feature_uses_last_30_days": 3,
        "support_requests_last_30_days": 4,
        "days_since_last_activity": 18,
    }

    result = predict_churn(model, features)

    assert set(result) == {
        "churn_probability",
        "risk_level",
        "predicted_churn",
        "model_notice",
    }


def test_prediction_values_are_valid():
    model = load_model()

    features = {
        "plan": "pro",
        "monthly_price": 49.99,
        "tenure_months": 12,
        "logins_last_30_days": 10,
        "feature_uses_last_30_days": 20,
        "support_requests_last_30_days": 1,
        "days_since_last_activity": 3,
    }

    result = predict_churn(model, features)

    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_level"] in {"low", "medium", "high"}
    assert isinstance(result["predicted_churn"], bool)
    assert "synthetic data" in result["model_notice"]
