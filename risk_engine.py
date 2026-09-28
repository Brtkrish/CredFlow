"""
Risk engine: loads the trained model/preprocessor and turns a raw applicant
dict into a risk score + ranked feature contributions (via SHAP), with no
LLM involved. This keeps every number in the eventual explanation grounded
in an actual model computation rather than something an LLM invents.
"""
import joblib
import numpy as np
import pandas as pd
import shap

MODEL = joblib.load("model.joblib")
PREPROCESSOR = joblib.load("preprocessor.joblib")
FEATURE_NAMES = joblib.load("feature_names.joblib")
DECISION_THRESHOLD = joblib.load("threshold.joblib")
EXPLAINER = shap.TreeExplainer(MODEL)

NUMERIC_FEATURES = ["age", "credit_amount", "duration_months", "payment_to_income_ratio", "existing_credits"]
CATEGORICAL_FEATURES = ["job", "housing", "purpose", "savings_status", "employment_years"]
REQUIRED_FIELDS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _risk_band(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.6:
        return "medium"
    return "high"


def predict(applicant: dict, top_n: int = 3) -> dict:
    missing = [f for f in REQUIRED_FIELDS if f not in applicant]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    row = pd.DataFrame([{f: applicant[f] for f in REQUIRED_FIELDS}])
    encoded = PREPROCESSOR.transform(row)
    # OneHotEncoder output may be sparse; densify for SHAP
    encoded_dense = encoded.toarray() if hasattr(encoded, "toarray") else encoded

    risk_score = float(MODEL.predict_proba(encoded_dense)[0][1])  # P(bad)
    decision = "rejected" if risk_score >= DECISION_THRESHOLD else "approved"

    shap_values = EXPLAINER.shap_values(encoded_dense)
    # shap_values shape: (1, n_features) for binary XGBoost
    contributions = shap_values[0] if shap_values.ndim == 2 else shap_values[0][:, 1]

    ranked = sorted(
        zip(FEATURE_NAMES, contributions, encoded_dense[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    top_factors = [
        {
            "feature": name,
            "impact": "negative" if val > 0 else "positive",  # positive SHAP -> pushes toward "bad"
            "shap_value": round(float(val), 4),
        }
        for name, val, _ in ranked
    ]

    return {
        "decision": decision,
        "risk_score": round(risk_score, 4),
        "risk_band": _risk_band(risk_score),
        "top_factors": top_factors,
    }


if __name__ == "__main__":
    sample = {
        "age": 28,
        "job": "skilled employee/ official",
        "housing": "rent",
        "credit_amount": 6000,
        "duration_months": 36,
        "purpose": "car (new)",
        "savings_status": "< 100 DM",
        "employment_years": "1 to < 4 years",
        "payment_to_income_ratio": 4,
        "existing_credits": 1,
    }
    result = predict(sample)
    import json
    print(json.dumps(result, indent=2))
