
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from datetime import date, datetime
from decimal import Decimal
from typing import Literal




class SubscriptionCreate(BaseModel):
    customer_id: int = Field(gt=0)
    plan: Literal["basic", "pro", "enterprise"]
    monthly_price: Decimal = Field(gt=0, max_digits=10, decimal_places=2)
    start_date: date
    status: Literal["active", "cancelled", "past_due"] = "active"


class SubscriptionResponse(BaseModel):
    id: int
    customer_id: int
    plan: str
    monthly_price: Decimal
    start_date: date
    status: str

    model_config = ConfigDict(from_attributes=True)



class CustomerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr


class CustomerResponse(BaseModel):
    id: int
    name: str
    email: EmailStr

    model_config = ConfigDict(from_attributes=True)



class UsageEventCreate(BaseModel):
    customer_id: int = Field(gt=0)
    event_type: Literal[
        "login",
        "feature_used",
        "support_request",
    ]
    description: str | None = Field(
        default=None,
        max_length=2000,
    )


class UsageEventResponse(BaseModel):
    id: int
    customer_id: int
    event_type: str
    description: str | None
    occurred_at: datetime

    model_config = ConfigDict(from_attributes=True)
