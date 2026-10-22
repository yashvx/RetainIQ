
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Customer, Subscription, UsageEvent
from ml.predict import load_model, predict_churn

from app.schemas import (
    CustomerCreate,
    CustomerResponse,
    SubscriptionCreate,
    SubscriptionResponse,
    UsageEventCreate,
    UsageEventResponse,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
)


tags_metadata = [
    {
        "name": "Machine Learning",
        "description": "Prototype customer churn predictions.",
    },
    {
        "name": "Analytics",
        "description": "Customer activity metrics and insights.",
    },
    {
        "name": "System Health",
        "description": "Check whether the RetainIQ API is running.",
    },
    {
        "name": "Customers",
        "description": "Create and retrieve customer records.",
    },
    {
        "name": "Subscriptions",
        "description": "Manage customer subscription plans and billing details.",
    },
    {
        "name": "Usage Events",
        "description": "Record and retrieve customer activity.",
    },
]

app = FastAPI(
    title="RetainIQ API",
    description="Customer churn prediction and retention platform",
    version="0.1.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)


churn_model = load_model()


@app.get("/")
def root():
    return {
        "message": "Welcome to RetainIQ",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post(
    "/predict/churn",
    response_model=ChurnPredictionResponse,
    tags=["Machine Learning"],
)
def predict_customer_churn(
    request: ChurnPredictionRequest,
):
    features = request.model_dump()
    return predict_churn(churn_model, features)


@app.post(
    "/customers",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Customers"],
)
def create_customer(
    customer_data: CustomerCreate,
    db: Session = Depends(get_db),
):
    customer = Customer(
        name=customer_data.name,
        email=str(customer_data.email),
    )
    db.add(customer)

    try:
        db.commit()
        db.refresh(customer)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A customer with this email already exists.",
        )

    return customer


@app.get("/customers", response_model=list[CustomerResponse])
def list_customers(db: Session = Depends(get_db)):
    statement = select(Customer).order_by(Customer.id)
    return db.scalars(statement).all()


@app.post(
    "/subscriptions",
    response_model=SubscriptionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Subscriptions"],
)
def create_subscription(
    subscription_data: SubscriptionCreate,
    db: Session = Depends(get_db),
):
    customer = db.get(Customer, subscription_data.customer_id)

    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )

    subscription = Subscription(**subscription_data.model_dump())
    db.add(subscription)
    db.commit()
    db.refresh(subscription)

    return subscription


@app.get(
    "/subscriptions",
    response_model=list[SubscriptionResponse],
)
def list_subscriptions(db: Session = Depends(get_db)):
    statement = select(Subscription).order_by(Subscription.id)
    return db.scalars(statement).all()


@app.post(
    "/usage-events",
    response_model=UsageEventResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Usage Events"],
)
def create_usage_event(
    event_data: UsageEventCreate,
    db: Session = Depends(get_db),
):
    customer = db.get(Customer, event_data.customer_id)

    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )

    event = UsageEvent(**event_data.model_dump())
    db.add(event)
    db.commit()
    db.refresh(event)

    return event


@app.get(
    "/usage-events",
    response_model=list[UsageEventResponse],
)
def list_usage_events(
    customer_id: int | None = None,
    db: Session = Depends(get_db),
):
    statement = select(UsageEvent).order_by(UsageEvent.id)

    if customer_id is not None:
        statement = statement.where(UsageEvent.customer_id == customer_id)

    return db.scalars(statement).all()


@app.get(
    "/customers/{customer_id}/predict-churn",
    response_model=ChurnPredictionResponse,
    tags=["Machine Learning"],
)
def predict_customer_churn_from_database(
    customer_id: int,
    db: Session = Depends(get_db),
):
    customer = db.get(Customer, customer_id)

    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )

    subscription_statement = (
        select(Subscription)
        .where(Subscription.customer_id == customer_id)
        .order_by(Subscription.id.desc())
    )
    subscription = db.scalars(subscription_statement).first()

    if subscription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No subscription found for this customer.",
        )

    now = datetime.now(timezone.utc)
    thirty_days_ago = now - timedelta(days=30)

    events_statement = (
        select(UsageEvent)
        .where(UsageEvent.customer_id == customer_id)
        .order_by(UsageEvent.occurred_at)
    )
    events = db.scalars(events_statement).all()

    def as_utc(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    recent_events = [
        event for event in events
        if as_utc(event.occurred_at) >= thirty_days_ago
    ]

    last_activity = (
        max(as_utc(event.occurred_at) for event in events)
        if events
        else None
    )

    days_since_last_activity = (
        max(0, (now - last_activity).days)
        if last_activity is not None
        else 365
    )

    features = {
        "plan": subscription.plan,
        "monthly_price": float(subscription.monthly_price),
        "tenure_months": max(
            1,
            (now.date() - subscription.start_date).days // 30,
        ),
        "logins_last_30_days": sum(
            event.event_type == "login" for event in recent_events
        ),
        "feature_uses_last_30_days": sum(
            event.event_type == "feature_used" for event in recent_events
        ),
        "support_requests_last_30_days": sum(
            event.event_type == "support_request" for event in recent_events
        ),
        "days_since_last_activity": min(days_since_last_activity, 3650),
    }

    result = predict_churn(churn_model, features)

    return {
        **result,
        "customer_id": customer_id,
        "features_used": features,
    }


@app.get(
    "/analytics/customers/{customer_id}/activity",
    tags=["Analytics"],
)
def get_customer_activity(
    customer_id: int,
    db: Session = Depends(get_db),
):
    customer = db.get(Customer, customer_id)

    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )

    now = datetime.now(timezone.utc)
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    statement = (
        select(UsageEvent)
        .where(UsageEvent.customer_id == customer_id)
        .order_by(UsageEvent.occurred_at)
    )
    events = db.scalars(statement).all()

    def as_utc(timestamp: datetime) -> datetime:
        # Support older timestamps that may lack timezone information.
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    recent_events = [
        event
        for event in events
        if as_utc(event.occurred_at) >= seven_days_ago
    ]

    monthly_events = [
        event
        for event in events
        if as_utc(event.occurred_at) >= thirty_days_ago
    ]

    login_count = sum(event.event_type == "login" for event in events)
    feature_used_count = sum(event.event_type == "feature_used" for event in events)
    support_request_count = sum(
        event.event_type == "support_request" for event in events
    )

    recent_login_count = sum(event.event_type == "login" for event in recent_events)

    last_activity = (
        max(as_utc(event.occurred_at) for event in events)
        if events
        else None
    )

    days_since_last_activity = (
        max(0, (now - last_activity).days)
        if last_activity is not None
        else None
    )

    return {
        "customer_id": customer_id,
        "total_events": len(events),
        "login_count": login_count,
        "feature_used_count": feature_used_count,
        "support_request_count": support_request_count,
        "events_last_7_days": len(recent_events),
        "events_last_30_days": len(monthly_events),
        "logins_last_7_days": recent_login_count,
        "last_activity_at": (
            last_activity.isoformat()
            if last_activity is not None
            else None
        ),
        "days_since_last_activity": days_since_last_activity,
    }
