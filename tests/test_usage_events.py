
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

    db = TestingSessionLocal()
    db.add_all([
        Customer(name="Alice", email="alice-events@example.com"),
        Customer(name="Bob", email="bob-events@example.com"),
    ])
    db.commit()
    db.close()

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=test_engine)


def test_create_usage_event(client):
    response = client.post(
        "/usage-events",
        json={
            "customer_id": 1,
            "event_type": "login",
            "description": "Logged in to dashboard",
        },
    )

    assert response.status_code == 201

    data = response.json()
    assert data["customer_id"] == 1
    assert data["event_type"] == "login"
    assert data["description"] == "Logged in to dashboard"
    assert data["id"] > 0
    assert data["occurred_at"]


def test_reject_event_for_missing_customer(client):
    response = client.post(
        "/usage-events",
        json={
            "customer_id": 999,
            "event_type": "login",
            "description": "Unknown customer",
        },
    )

    assert response.status_code == 404


def test_reject_invalid_event_type(client):
    response = client.post(
        "/usage-events",
        json={
            "customer_id": 1,
            "event_type": "watched_movie",
            "description": "Unsupported event",
        },
    )

    assert response.status_code == 422


def test_filter_usage_events_by_customer(client):
    client.post(
        "/usage-events",
        json={
            "customer_id": 1,
            "event_type": "login",
            "description": "Alice logged in",
        },
    )

    client.post(
        "/usage-events",
        json={
            "customer_id": 2,
            "event_type": "feature_used",
            "description": "Bob used analytics",
        },
    )

    response = client.get("/usage-events?customer_id=1")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["customer_id"] == 1


