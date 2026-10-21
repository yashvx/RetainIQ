
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.main import app
from app.models import Base, Customer


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

    # Each test gets a customer to use for subscription creation.
    db = TestingSessionLocal()
    db.add(
        Customer(
            name="Test Customer",
            email="subscription-test@example.com",
        )
    )
    db.commit()
    db.close()

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=test_engine)


def test_create_subscription(client):
    response = client.post(
        "/subscriptions",
        json={
            "customer_id": 1,
            "plan": "pro",
            "monthly_price": "999.00",
            "start_date": "2026-10-21",
            "status": "active",
        },
    )

    assert response.status_code == 201

    data = response.json()
    assert data["customer_id"] == 1
    assert data["plan"] == "pro"
    assert Decimal(data["monthly_price"]) == Decimal("999.00")
    assert data["start_date"] == "2026-10-21"
    assert data["status"] == "active"


def test_reject_subscription_for_missing_customer(client):
    response = client.post(
        "/subscriptions",
        json={
            "customer_id": 999,
            "plan": "pro",
            "monthly_price": "999.00",
            "start_date": "2026-10-21",
            "status": "active",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Customer not found."


def test_reject_invalid_plan(client):
    response = client.post(
        "/subscriptions",
        json={
            "customer_id": 1,
            "plan": "premium",
            "monthly_price": "999.00",
            "start_date": "2026-10-21",
            "status": "active",
        },
    )

    assert response.status_code == 422


def test_reject_negative_monthly_price(client):
    response = client.post(
        "/subscriptions",
        json={
            "customer_id": 1,
            "plan": "pro",
            "monthly_price": "-100.00",
            "start_date": "2026-10-21",
            "status": "active",
        },
    )

    assert response.status_code == 422
