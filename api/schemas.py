"""Schémas Pydantic pour l'API de prédiction de churn et estimation CLV."""

from pydantic import BaseModel, Field
from typing import Optional, List


class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: str = Field(..., description="Male ou Female")
    country: str = Field(default="UK")
    city: str = Field(default="London")
    customer_segment: str = Field(..., description="Individual, SME ou Enterprise")
    tenure_months: int = Field(..., ge=0)
    signup_channel: str = Field(..., description="Web, Mobile ou Referral")
    contract_type: str = Field(..., description="Monthly, Quarterly ou Yearly")
    monthly_logins: int = Field(..., ge=0)
    weekly_active_days: int = Field(..., ge=0, le=7)
    avg_session_time: float = Field(..., ge=0)
    features_used: int = Field(..., ge=0)
    usage_growth_rate: float = Field(default=0.0)
    last_login_days_ago: int = Field(..., ge=0)
    monthly_fee: float = Field(..., ge=0)
    total_revenue: float = Field(..., ge=0)
    payment_method: str = Field(..., description="Card, PayPal ou Bank Transfer")
    payment_failures: int = Field(..., ge=0)
    discount_applied: str = Field(default="No", description="Yes ou No")
    price_increase_last_3m: str = Field(default="No", description="Yes ou No")
    support_tickets: int = Field(..., ge=0)
    avg_resolution_time: float = Field(default=0.0, ge=0)
    complaint_type: str = Field(default="None", description="None, Billing, Service ou Technical")
    csat_score: float = Field(..., ge=0, le=5)
    escalations: int = Field(default=0, ge=0)
    email_open_rate: float = Field(default=0.5, ge=0, le=1)
    marketing_click_rate: float = Field(default=0.2, ge=0, le=1)
    nps_score: float = Field(..., ge=0, le=100)
    survey_response: str = Field(default="Neutral", description="Satisfied, Neutral ou Unsatisfied")
    referral_count: int = Field(default=0, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "gender": "Male",
                "country": "France",
                "city": "Paris",
                "customer_segment": "SME",
                "tenure_months": 12,
                "signup_channel": "Web",
                "contract_type": "Monthly",
                "monthly_logins": 15,
                "weekly_active_days": 4,
                "avg_session_time": 25.5,
                "features_used": 5,
                "usage_growth_rate": 0.05,
                "last_login_days_ago": 3,
                "monthly_fee": 79.99,
                "total_revenue": 959.88,
                "payment_method": "Card",
                "payment_failures": 2,
                "discount_applied": "No",
                "price_increase_last_3m": "No",
                "support_tickets": 3,
                "avg_resolution_time": 10.0,
                "complaint_type": "None",
                "csat_score": 3.5,
                "escalations": 0,
                "email_open_rate": 0.6,
                "marketing_click_rate": 0.3,
                "nps_score": 40.0,
                "survey_response": "Neutral",
                "referral_count": 1
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    revenue_at_risk: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    total_revenue_at_risk: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    features: List[str]
    categorical_features: List[str]
    numerical_features: List[str]
    threshold: float
    description: str
