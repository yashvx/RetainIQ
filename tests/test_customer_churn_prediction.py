from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.main import app
from app.models import Base, Customer, Subscription, UsageEvent


test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    autocommit=False,
)


@pytest.fixture
def client():
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    db = TestingSessionLocal()
    db.add_all([
        Customer(name="Alice", email="alice-churn@example.com"),
        Customer(name="Bob", email="bob-churn@example.com"),
        Customer(name="Charlie", email="charlie-churn@example.com"),
    ])
    db.commit()

    db.add_all([
        Subscription(
            customer_id=1,
            plan="basic",
            monthly_price=Decimal("29.99"),
            start_date=date.today() - timedelta(days=120),
            status="active",
        ),
        Subscription(
            customer_id=2,
            plan="pro",
            monthly_price=Decimal("49.99"),
            start_date=date.today() - timedelta(days=365),
            status="active",
        ),
    ])
    db.add_all([
        UsageEvent(
            customer_id=1,
            event_type="login",
            description="Recent login",
            occurred_at=datetime.now(timezone.utc) - timedelta(days=1),
        ),
        UsageEvent(
            customer_id=1,
            event_type="feature_used",
            description="Used a feature",
            occurred_at=datetime.now(timezone.utc) - timedelta(days=2),
        ),
        UsageEvent(
            customer_id=1,
            event_type="support_request",
            description="Requested support",
            occurred_at=datetime.now(timezone.utc) - timedelta(days=3),
        ),
    ])
    db.commit()
    db.close()

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=test_engine)


def test_predict_churn_for_customer_with_subscription_and_events(client):
    response = client.get("/customers/1/predict-churn")

    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == 1
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_level"] in {"low", "medium", "high"}
    assert data["predicted_churn"] in {True, False}
    assert data["features_used"]["plan"] == "basic"
    assert data["features_used"]["logins_last_30_days"] == 1
    assert data["features_used"]["feature_uses_last_30_days"] == 1
    assert data["features_used"]["support_requests_last_30_days"] == 1
    assert "synthetic data" in data["model_notice"]


def test_predict_churn_for_customer_without_usage_events(client):
    response = client.get("/customers/2/predict-churn")

    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == 2
    assert data["features_used"]["logins_last_30_days"] == 0
    assert data["features_used"]["feature_uses_last_30_days"] == 0
    assert data["features_used"]["support_requests_last_30_days"] == 0
    assert data["features_used"]["days_since_last_activity"] == 365


def test_predict_churn_for_customer_without_subscription_returns_404(client):
    response = client.get("/customers/3/predict-churn")

    assert response.status_code == 404
    assert response.json()["detail"] == "No subscription found for this customer."


def test_predict_churn_for_missing_customer_returns_404(client):
    response = client.get("/customers/999/predict-churn")

    assert response.status_code == 404
    assert response.json()["detail"] == "Customer not found."
