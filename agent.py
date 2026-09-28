"""
Agentic explainability layer, built as a LangGraph state graph with two
real nodes:

  1. retrieve  -- TF-IDF similarity search over policy_docs.py, using the
                  applicant's actual top SHAP factors as the query, so the
                  agent pulls in genuinely relevant policy text (not a
                  hardcoded snippet keyed by feature name).
  2. generate  -- calls Groq's LLM API (OpenAI-compatible, free tier) with
                  the risk engine's numbers *and* the retrieved policy
                  context, and asks it to write the advisory note grounded
                  in both.

This replaces the single direct LLM call in the original explain.py with
an actual multi-step, retrieval-augmented agent.
"""
import json
import os
from typing import TypedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, END

from policy_docs import POLICY_DOCS

# --- Retrieval setup (built once at import time) ---------------------------
_DOC_TEXTS = [d["text"] for d in POLICY_DOCS]
_VECTORIZER = TfidfVectorizer(stop_words="english")
_DOC_MATRIX = _VECTORIZER.fit_transform(_DOC_TEXTS)


def retrieve_policy_snippets(top_factors: list[dict], k: int = 3) -> list[dict]:
    """Retrieve the most relevant policy docs for this prediction's top
    SHAP factors, using both tag matching and TF-IDF similarity as a
    fallback for factors that don't exactly match a tag (e.g. one-hot
    encoded feature names like 'savings_status_< 100 DM')."""
    query = " ".join(f["feature"] for f in top_factors)

    # exact/prefix tag matches first
    matched = [
        d for d in POLICY_DOCS
        if any(query_feature.startswith(tag) or tag in query_feature
               for tag in d["tags"] for query_feature in [f["feature"] for f in top_factors])
    ]

    if len(matched) >= k:
        return matched[:k]

    # fill remaining slots via TF-IDF similarity on the query text
    query_vec = _VECTORIZER.transform([query])
    sims = cosine_similarity(query_vec, _DOC_MATRIX)[0]
    ranked_idx = sims.argsort()[::-1]
    for idx in ranked_idx:
        doc = POLICY_DOCS[idx]
        if doc not in matched:
            matched.append(doc)
        if len(matched) >= k:
            break
    return matched[:k]


# --- LangGraph state + nodes -------------------------------------------------
class AgentState(TypedDict):
    applicant: dict
    prediction: dict
    retrieved_docs: list[dict]
    explanation: str


def retrieve_node(state: AgentState) -> AgentState:
    docs = retrieve_policy_snippets(state["prediction"]["top_factors"])
    return {**state, "retrieved_docs": docs}


def generate_node(state: AgentState) -> AgentState:
    api_key = os.environ.get("GROQ_API_KEY")
    prediction = state["prediction"]
    applicant = state["applicant"]
    docs = state["retrieved_docs"]

    if not api_key:
        state["explanation"] = _template_fallback(prediction, docs)
        return state

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    policy_context = "\n".join(f"- {d['text']}" for d in docs)
    prompt = f"""You are a loan advisory assistant. Given this credit decision and the
relevant internal policy guidance below, write a 2-3 sentence advisory note
for the applicant in plain English. Only use the numbers provided -- do not
invent or estimate any figures. If rejected, ground your one concrete
suggestion in the retrieved policy guidance, not a generic tip.

Decision: {prediction['decision']}
Risk score: {prediction['risk_score']} ({prediction['risk_band']} risk)
Top contributing factors: {json.dumps(prediction['top_factors'])}
Applicant loan amount: {applicant.get('credit_amount')}
Applicant loan duration (months): {applicant.get('duration_months')}

Retrieved policy guidance:
{policy_context}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    state["explanation"] = response.choices[0].message.content
    return state


def _template_fallback(prediction: dict, docs: list[dict]) -> str:
    parts = [
        f"Loan {prediction['decision']} with a risk score of "
        f"{prediction['risk_score']:.2f} ({prediction['risk_band']} risk)."
    ]
    if docs:
        parts.append(f"Relevant guidance: {docs[0]['text']}")
    return " ".join(parts)


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_AGENT = build_graph()


def build_explanation(applicant: dict, prediction: dict) -> dict:
    """Runs the retrieve -> generate agent and returns both the explanation
    and which policy docs were retrieved (useful for the UI/audit trail)."""
    result = _AGENT.invoke({
        "applicant": applicant,
        "prediction": prediction,
        "retrieved_docs": [],
        "explanation": "",
    })
    return {
        "explanation": result["explanation"],
        "retrieved_policy_ids": [d["id"] for d in result["retrieved_docs"]],
    }


if __name__ == "__main__":
    sample_prediction = {
        "decision": "rejected",
        "risk_score": 0.81,
        "risk_band": "high",
        "top_factors": [
            {"feature": "duration_months", "impact": "negative", "shap_value": 0.65},
            {"feature": "credit_amount", "impact": "positive", "shap_value": -0.54},
            {"feature": "housing_own", "impact": "negative", "shap_value": 0.26},
        ],
    }
    sample_applicant = {"credit_amount": 6000, "duration_months": 36}
    out = build_explanation(sample_applicant, sample_prediction)
    print(json.dumps(out, indent=2))
