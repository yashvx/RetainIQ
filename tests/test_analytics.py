from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.main import app
from app.models import Base, Customer, UsageEvent


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
        Customer(name="Alice", email="alice-analytics@example.com"),
        Customer(name="Bob", email="bob-analytics@example.com"),
    ])
    db.commit()

    db.add_all([
        UsageEvent(
            customer_id=1,
            event_type="login",
            description="First login",
        ),
        UsageEvent(
            customer_id=1,
            event_type="login",
            description="Second login",
        ),
        UsageEvent(
            customer_id=1,
            event_type="feature_used",
            description="Used analytics",
        ),
        UsageEvent(
            customer_id=1,
            event_type="support_request",
            description="Requested help",
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


def test_activity_summary_counts_events_correctly(client):
    response = client.get("/analytics/customers/1/activity")

    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == 1
    assert data["total_events"] == 4
    assert data["login_count"] == 2
    assert data["feature_used_count"] == 1
    assert data["support_request_count"] == 1
    assert data["events_last_7_days"] == 4
    assert data["events_last_30_days"] == 4
    assert data["logins_last_7_days"] == 2
    assert data["last_activity_at"] is not None
    assert data["days_since_last_activity"] == 0


def test_activity_summary_for_customer_with_no_events(client):
    response = client.get("/analytics/customers/2/activity")

    assert response.status_code == 200
    assert response.json() == {
        "customer_id": 2,
        "total_events": 0,
        "login_count": 0,
        "feature_used_count": 0,
        "support_request_count": 0,
        "events_last_7_days": 0,
        "events_last_30_days": 0,
        "logins_last_7_days": 0,
        "last_activity_at": None,
        "days_since_last_activity": None,
    }


def test_activity_summary_for_missing_customer(client):
    response = client.get("/analytics/customers/999/activity")

    assert response.status_code == 404
    assert response.json()["detail"] == "Customer not found."
