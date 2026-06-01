"""
API REST FastAPI — Prédiction de churn + estimation CLV.
Lancement : uvicorn api.main:app --reload --port 8000
Docs      : http://localhost:8000/docs
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    CustomerFeatures, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse
)

# --- Configuration ---

app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction de churn client et estimation CLV — M2 Data Science",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_VERSION = "1.0.0"
THRESHOLD = 0.5
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

NUMERICAL_FEATURES = [
    'age', 'tenure_months', 'monthly_logins', 'weekly_active_days',
    'avg_session_time', 'features_used', 'usage_growth_rate',
    'last_login_days_ago', 'monthly_fee', 'payment_failures',
    'support_tickets', 'avg_resolution_time', 'csat_score',
    'escalations', 'email_open_rate', 'marketing_click_rate',
    'nps_score', 'referral_count'
]

CATEGORICAL_FEATURES = [
    'gender', 'country', 'city', 'customer_segment', 'signup_channel',
    'contract_type', 'payment_method', 'discount_applied',
    'price_increase_last_3m', 'complaint_type', 'survey_response'
]

# Chargement du modèle au démarrage
model = None
try:
    path = os.path.join(MODELS_DIR, 'best_model.joblib')
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"Modèle chargé : {path}")
    else:
        print(f"Modèle introuvable à {path}. Lancez python src/train.py.")
except Exception as e:
    print(f"Erreur chargement modèle : {e}")


def customer_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame([{
        'age': customer.age,
        'gender': customer.gender,
        'country': customer.country,
        'city': customer.city,
        'customer_segment': customer.customer_segment,
        'tenure_months': customer.tenure_months,
        'signup_channel': customer.signup_channel,
        'contract_type': customer.contract_type,
        'monthly_logins': customer.monthly_logins,
        'weekly_active_days': customer.weekly_active_days,
        'avg_session_time': customer.avg_session_time,
        'features_used': customer.features_used,
        'usage_growth_rate': customer.usage_growth_rate,
        'last_login_days_ago': customer.last_login_days_ago,
        'monthly_fee': customer.monthly_fee,
        'total_revenue': customer.total_revenue,
        'payment_method': customer.payment_method,
        'payment_failures': customer.payment_failures,
        'discount_applied': customer.discount_applied,
        'price_increase_last_3m': customer.price_increase_last_3m,
        'support_tickets': customer.support_tickets,
        'avg_resolution_time': customer.avg_resolution_time,
        'complaint_type': customer.complaint_type,
        'csat_score': customer.csat_score,
        'escalations': customer.escalations,
        'email_open_rate': customer.email_open_rate,
        'marketing_click_rate': customer.marketing_click_rate,
        'nps_score': customer.nps_score,
        'survey_response': customer.survey_response,
        'referral_count': customer.referral_count
    }])


def proba_to_risk_level(proba: float) -> str:
    if proba < 0.3:
        return "Low"
    elif proba < 0.7:
        return "Medium"
    return "High"


def predict_single(customer: CustomerFeatures) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible. Lancez python src/train.py.")
    try:
        df = customer_to_dataframe(customer)
        proba = float(model.predict_proba(df)[0, 1])
        prediction = int(proba >= THRESHOLD)
        revenue_at_risk = customer.monthly_fee * proba

        return PredictionResponse(
            churn_probability=round(proba, 4),
            churn_prediction=prediction,
            risk_level=proba_to_risk_level(proba),
            revenue_at_risk=round(revenue_at_risk, 2),
            model_version=MODEL_VERSION
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Vérifie l'état du service."""
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=MODEL_VERSION
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    """Métadonnées du modèle déployé."""
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        features=NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        numerical_features=NUMERICAL_FEATURES,
        threshold=THRESHOLD,
        description="Modèle de prédiction de churn entraîné sur customer_churn_business_dataset.csv (10 000 clients)."
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(customer: CustomerFeatures):
    """
    Prédit la probabilité de churn pour un client.

    - **churn_probability** : 0 à 1
    - **risk_level** : Low / Medium / High
    - **revenue_at_risk** : monthly_fee × churn_probability
    """
    return predict_single(customer)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prédiction"])
def predict_batch(request: BatchPredictionRequest):
    """Prédictions pour une liste de clients (max 1 000)."""
    if len(request.customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch limité à 1 000 clients.")

    predictions = [predict_single(c) for c in request.customers]
    high_risk = sum(1 for p in predictions if p.risk_level == "High")
    total_rev = sum(p.revenue_at_risk for p in predictions)

    return BatchPredictionResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=high_risk,
        total_revenue_at_risk=round(total_rev, 2)
    )
