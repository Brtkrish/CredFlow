"""
A small internal 'lending policy' knowledge base -- short guidance snippets
tied to specific risk factors. This is what the retrieval step searches over,
so the agent's advisory notes are grounded in actual policy text rather than
whatever the LLM decides to say.

In a real fintech setting this would be the bank's actual underwriting
guidelines; here it's a compact stand-in that's still genuinely retrieved
and cited, not hardcoded per-response.
"""

POLICY_DOCS = [
    {
        "id": "duration_months",
        "tags": ["duration_months"],
        "text": (
            "Loan duration policy: longer repayment terms (over 24 months) carry "
            "higher default risk due to extended exposure to the applicant's "
            "changing financial circumstances. Recommended mitigation: shorten "
            "the requested term or require a co-signer for terms beyond 36 months."
        ),
    },
    {
        "id": "credit_amount",
        "tags": ["credit_amount"],
        "text": (
            "Loan amount policy: requests above 5,000 relative to the applicant's "
            "profile increase exposure. Recommended mitigation: reduce the "
            "principal requested, or provide additional collateral."
        ),
    },
    {
        "id": "savings_status",
        "tags": ["savings_status"],
        "text": (
            "Savings policy: applicants with little to no savings buffer have "
            "reduced capacity to absorb income shocks during the loan term. "
            "Recommended mitigation: pair with a smaller loan amount or a shorter term."
        ),
    },
    {
        "id": "payment_to_income_ratio",
        "tags": ["payment_to_income_ratio"],
        "text": (
            "Debt-to-income policy: an installment burden above 3 (on a 1-4 scale) "
            "signals limited disposable income relative to obligations. "
            "Recommended mitigation: lower the loan amount so the installment "
            "share of income decreases, or extend the term modestly."
        ),
    },
    {
        "id": "employment_years",
        "tags": ["employment_years"],
        "text": (
            "Employment stability policy: applicants employed under 1 year have "
            "less verifiable income history. Recommended mitigation: request "
            "additional income verification or a guarantor."
        ),
    },
    {
        "id": "housing",
        "tags": ["housing_own", "housing_rent", "housing_for free"],
        "text": (
            "Housing stability policy: applicants who do not own their residence "
            "are assessed with slightly higher scrutiny on residual monthly "
            "obligations, since rent is an additional fixed cost competing with "
            "loan repayment."
        ),
    },
    {
        "id": "existing_credits",
        "tags": ["existing_credits"],
        "text": (
            "Existing obligations policy: multiple concurrent credit lines "
            "increase aggregate repayment obligations. Recommended mitigation: "
            "confirm total monthly debt service across all lines before approving "
            "additional credit."
        ),
    },
    {
        "id": "age",
        "tags": ["age"],
        "text": (
            "Applicant age is considered only insofar as it correlates with "
            "employment and income stability, per fair-lending guidelines; it is "
            "not itself grounds for adverse action."
        ),
    },
]
