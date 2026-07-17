"""
FastAPI inference service for the fraud detection pipeline.

Run:
    uvicorn src.api:app --reload --port 8000

Then POST to /predict, e.g.:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
      "claim_amount": 9800, "num_prev_claims": 1, "days_since_policy": 40,
      "location": "urban", "disaster_nearby": 0, "num_payments": 1,
      "payment_irregularity": 1, "narrative": "Total loss claimed, no documentation provided.",
      "image_path": "data/image_folder/img_0.png"
    }'
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_pipeline.pkl")

app = FastAPI(
    title="Insurance Fraud Detection API",
    description="Multimodal (structured + text + image) fraud risk scoring for insurance claims.",
    version="1.0.0",
)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.",
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


class ClaimRequest(BaseModel):
    claim_amount: float = Field(..., gt=0, example=9800.0)
    num_prev_claims: int = Field(..., ge=0, example=1)
    days_since_policy: int = Field(..., ge=0, example=40)
    location: str = Field(..., example="urban")
    disaster_nearby: int = Field(..., ge=0, le=1, example=0)
    num_payments: int = Field(..., ge=0, example=1)
    payment_irregularity: int = Field(..., ge=0, le=1, example=1)
    narrative: str = Field(..., example="Total loss claimed, no documentation provided.")
    image_path: str = Field(
        ..., example="data/image_folder/img_0.png",
        description="Path to the claim image on disk (or a shared volume in production).",
    )


class ClaimResponse(BaseModel):
    fraud_probability: float
    is_fraud_predicted: bool
    threshold_used: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict", response_model=ClaimResponse)
def predict(claim: ClaimRequest, threshold: float = 0.5):
    pipe = get_pipeline()
    row = pd.DataFrame([claim.dict()])
    proba = float(pipe.predict_proba(row)[:, 1][0])
    return ClaimResponse(
        fraud_probability=round(proba, 4),
        is_fraud_predicted=proba >= threshold,
        threshold_used=threshold,
    )
