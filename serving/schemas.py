"""Request/response schemas for the serving API.

Categorical domains are validated against ``src.config.CATEGORICAL_DOMAINS`` —
the same single source of truth used by training and Great Expectations — so
the API can never silently accept values the model was not trained on.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src import config

MAX_BATCH_ROWS = 1000  # protects the 512 MB Render instance from giant payloads


class CustomerFeatures(BaseModel):
    """One customer, exactly the 19 training features. Unknown keys rejected."""

    model_config = {"extra": "forbid"}

    # numeric
    tenure: int = Field(ge=0, le=1000)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    SeniorCitizen: int

    # categorical (domains enforced below, not via Literal, so the allowed
    # values stay defined in src.config only)
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    @field_validator("SeniorCitizen")
    @classmethod
    def _senior_citizen_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("SeniorCitizen must be 0 or 1")
        return v

    @field_validator(*config.CATEGORICAL_DOMAINS.keys())
    @classmethod
    def _in_domain(cls, v: str, info):
        domain = config.CATEGORICAL_DOMAINS[info.field_name]
        if v not in domain:
            raise ValueError(f"{info.field_name} must be one of {domain}")
        return v


class Prediction(BaseModel):
    churn_probability: float
    churn: bool
    model_source: str


class BatchRequest(BaseModel):
    records: list[CustomerFeatures] = Field(min_length=1, max_length=MAX_BATCH_ROWS)


class BatchResponse(BaseModel):
    count: int
    predictions: list[Prediction]
