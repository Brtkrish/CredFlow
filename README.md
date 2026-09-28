# Intelligent Credit Scoring & Explainable Loan Decision Agent

An end-to-end fintech backend system that evaluates credit risk using a
supervised ML model and produces transparent, natural-language reasoning
for every approval/rejection decision.

## Why this exists

Most credit-scoring demos stop at "here's a probability." In lending,
that's not enough -- an applicant (or a loan officer) needs to know *why*,
and ideally what would need to change. This project pairs a risk model
with an explainability layer that narrates the model's own SHAP-based
feature attributions in plain English, so nothing in the explanation is
invented -- it's grounded in the actual computation.

## Architecture

```
applicant data --> [risk_engine.py] --> risk score + SHAP top factors
                                              |
                                              v
                              [agent.py: LangGraph agent]
                              retrieve (TF-IDF over policy_docs.py)
                                              |
                                              v
                              generate (Claude, grounded in retrieved
                                         policy text + SHAP numbers)
                                              |
                                              v
                              [main.py: FastAPI /predict-risk]  <-- API
                              [app.py: Streamlit UI]            <-- demo frontend
```

## Model

- **Dataset:** German Credit Data (UCI/Statlog, 1000 applicants, 20 attributes)
- **Preprocessing:** one-hot encoding for categoricals; log1p transform on
  right-skewed numeric fields (`credit_amount`, `duration_months`) to reduce
  outlier dominance in tree splits
- **Model:** XGBoost classifier, class-weighted (to reflect that missing a
  risky applicant is costlier than a false alarm on a good one, per the
  dataset's own published cost matrix) and hyperparameter-tuned via
  5-fold cross-validated random search
- **Metrics:** 5-fold CV ROC-AUC **0.7345** (see `metrics.json` for full
  holdout numbers and best params after running `train.py`); decision
  threshold separately tuned to maximize recall on "bad" risk
- **Explainability:** LangGraph agent with two nodes -- `retrieve` (TF-IDF
  similarity search over a small internal lending-policy knowledge base,
  `policy_docs.py`, keyed to the applicant's actual top SHAP factors) and
  `generate` (Claude call grounded in both the retrieved policy text and
  the model's numbers). Falls back to a templated, policy-citing note if
  no API key is set, so the retrieval step is still demoable without one.

## Setup

```bash
pip install -r requirements.txt
python train.py          # trains and saves model.joblib, preprocessor.joblib, etc.
export ANTHROPIC_API_KEY=your_key_here   # optional -- falls back to a template otherwise
```

## Run the API

```bash
uvicorn main:app --reload
# POST http://localhost:8000/predict-risk
```

## Run the demo UI

```bash
streamlit run app.py
```

## Example request

```json
POST /predict-risk
{
  "age": 28,
  "job": "skilled employee/ official",
  "housing": "rent",
  "credit_amount": 6000,
  "duration_months": 36,
  "purpose": "car (new)",
  "savings_status": "< 100 DM",
  "employment_years": "1 to < 4 years",
  "payment_to_income_ratio": 4,
  "existing_credits": 1
}
```

## Possible extensions

- PostgreSQL persistence for application history
- `/predict-risk-bulk` for batch evaluation
- LangGraph agent wrapping multiple tools (policy lookup, counter-offer generation)
