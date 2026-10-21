
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.main import app
from app.models import Base


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

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=test_engine)


def test_create_customer(client):
    response = client.post(
        "/customers",
        json={
            "name": "Alice Smith",
            "email": "alice@example.com",
        },
    )

    assert response.status_code == 201
    assert response.json()["name"] == "Alice Smith"
    assert response.json()["email"] == "alice@example.com"
    assert response.json()["id"] > 0


def test_list_customers(client):
    client.post(
        "/customers",
        json={
            "name": "Bob Smith",
            "email": "bob@example.com",
        },
    )

    response = client.get("/customers")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["name"] == "Bob Smith"


def test_reject_invalid_email(client):
    response = client.post(
        "/customers",
        json={
            "name": "Charlie Smith",
            "email": "not-an-email",
        },
    )

    assert response.status_code == 422


def test_reject_duplicate_email(client):
    customer = {
        "name": "Dana Smith",
        "email": "dana@example.com",
    }

    first = client.post("/customers", json=customer)
    second = client.post("/customers", json=customer)

    assert first.status_code == 201
    assert second.status_code == 409
