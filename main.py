from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from risk_engine import predict
from agent import build_explanation

app = FastAPI(
    title="Intelligent Credit Scoring & Explainable Loan Decision Agent",
    description="Predicts credit risk and generates a transparent, natural-language decision note.",
    version="0.1.0",
)


class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    job: str
    housing: Literal["own", "rent", "for free"]
    credit_amount: int = Field(..., gt=0)
    duration_months: int = Field(..., gt=0)
    purpose: str
    savings_status: str
    employment_years: str
    payment_to_income_ratio: int = Field(..., ge=1, le=4)
    existing_credits: int = Field(..., ge=1)


class TopFactor(BaseModel):
    feature: str
    impact: Literal["positive", "negative"]
    shap_value: float


class RiskResponse(BaseModel):
    decision: Literal["approved", "rejected"]
    risk_score: float
    risk_band: Literal["low", "medium", "high"]
    top_factors: list[TopFactor]
    explanation: str
    retrieved_policy_ids: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-risk", response_model=RiskResponse)
def predict_risk(application: LoanApplication):
    try:
        applicant = application.model_dump()
        prediction = predict(applicant)
        agent_result = build_explanation(applicant, prediction)
        return {**prediction, **agent_result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
