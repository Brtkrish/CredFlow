import streamlit as st
from risk_engine import predict
from agent import build_explanation

st.set_page_config(page_title="Credit Scoring & Loan Decision Agent", page_icon="\U0001F4B3")
st.title("\U0001F4B3 Intelligent Credit Scoring & Explainable Loan Decision Agent")
st.caption("ML risk model + AI-generated advisory notes -- every explanation is grounded in the model's own SHAP values, not an LLM guess.")

with st.form("applicant_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        job = st.selectbox("Job", [
            "unskilled - resident", "unemployed/ unskilled - non-resident",
            "skilled employee/ official", "management/ self-employed/highly qualified employee",
        ])
        housing = st.selectbox("Housing", ["own", "rent", "for free"])
        purpose = st.selectbox("Purpose", [
            "car (new)", "car (used)", "furniture/equipment", "radio/television",
            "domestic appliances", "repairs", "education", "business", "retraining", "others",
        ])
        existing_credits = st.number_input("Existing credits", min_value=1, max_value=4, value=1)
    with col2:
        credit_amount = st.number_input("Loan amount requested (DM)", min_value=100, value=5000, step=100)
        duration_months = st.number_input("Loan duration (months)", min_value=1, value=24)
        savings_status = st.selectbox("Savings status", [
            "unknown/ no savings account", "< 100 DM", "100 to < 500 DM", "500 to < 1000 DM", ">= 1000 DM",
        ])
        employment_years = st.selectbox("Employment length", [
            "unemployed", "< 1 year", "1 to < 4 years", "4 to < 7 years", ">= 7 years",
        ])
        payment_to_income_ratio = st.slider("Installment as % of disposable income (1=low, 4=high)", 1, 4, 2)

    submitted = st.form_submit_button("Evaluate Application")

if submitted:
    applicant = {
        "age": age, "job": job, "housing": housing, "credit_amount": credit_amount,
        "duration_months": duration_months, "purpose": purpose, "savings_status": savings_status,
        "employment_years": employment_years, "payment_to_income_ratio": payment_to_income_ratio,
        "existing_credits": existing_credits,
    }
    result = predict(applicant)
    agent_result = build_explanation(applicant, result)
    explanation = agent_result["explanation"]

    if result["decision"] == "approved":
        st.success(f"\u2705 Decision: **{result['decision'].upper()}**")
    else:
        st.error(f"\u274C Decision: **{result['decision'].upper()}**")

    c1, c2 = st.columns(2)
    c1.metric("Risk score", f"{result['risk_score']:.2f}")
    c2.metric("Risk band", result["risk_band"].capitalize())

    st.subheader("Top contributing factors")
    for f in result["top_factors"]:
        icon = "\U0001F534" if f["impact"] == "negative" else "\U0001F7E2"
        st.write(f"{icon} **{f['feature']}** (SHAP: {f['shap_value']:+.3f})")

    st.subheader("Advisory note")
    st.info(explanation)

    if agent_result["retrieved_policy_ids"]:
        st.caption(f"Grounded in retrieved policy guidance: {', '.join(agent_result['retrieved_policy_ids'])}")
