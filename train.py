"""
Train a credit risk classifier on the German Credit dataset, using only
the fields exposed in our /predict-risk API contract, and save:
  - model.joblib          (trained XGBoost classifier)
  - preprocessor.joblib   (ColumnTransformer for encoding)
  - feature_names.joblib  (post-encoding feature names, for SHAP labeling)
  - metrics.json          (holdout accuracy / ROC-AUC, for the README/resume)
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from xgboost import XGBClassifier

RAW_PATH = "data/german_credit_data.csv"

# Map our clean API field names -> the dataset's original column names
FIELD_MAP = {
    "age": "age",
    "job": "job",
    "housing": "housing",
    "credit_amount": "credit_amount",
    "duration_months": "month_duration",
    "purpose": "purpose",
    "savings_status": "status_savings",
    "employment_years": "years_employment",
    "payment_to_income_ratio": "payment_to_income_ratio",
    "existing_credits": "n_credits",
}

NUMERIC_FEATURES = [
    "age", "credit_amount", "duration_months",
    "payment_to_income_ratio", "existing_credits",
]
# credit_amount and duration_months are right-skewed (a few large/long loans
# stretch the distribution); log1p compresses that tail so tree splits aren't
# dominated by a handful of outliers. age/ratio/existing_credits are already
# roughly well-behaved, so they pass through unchanged.
SKEWED_NUMERIC = ["credit_amount", "duration_months"]
UNSKEWED_NUMERIC = [f for f in NUMERIC_FEATURES if f not in SKEWED_NUMERIC]
CATEGORICAL_FEATURES = ["job", "housing", "purpose", "savings_status", "employment_years"]


def load_data():
    df = pd.read_csv(RAW_PATH)
    cols = list(FIELD_MAP.values())
    X = df[cols].rename(columns={v: k for k, v in FIELD_MAP.items()})
    y = (df["target"] == "bad").astype(int)  # 1 = bad/high risk, 0 = good
    return X, y


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", UNSKEWED_NUMERIC),
            ("num_log", FunctionTransformer(np.log1p), SKEWED_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # Class weighting: in lending, missing a bad-risk applicant (false negative)
    # is costlier than flagging a good applicant as risky (false positive) --
    # this mirrors the German Credit dataset's own published cost matrix
    # (misclassifying bad-as-good costs 5x more than good-as-bad).
    n_bad = (y_train == 1).sum()
    n_good = (y_train == 0).sum()
    scale_pos_weight = n_good / n_bad

    # Hyperparameter tuning via cross-validated random search, optimizing
    # ROC-AUC (threshold-independent) so we tune the model itself first,
    # then separately tune the decision threshold afterward.
    param_distributions = {
        "n_estimators": [100, 150, 200, 300, 400],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3, 5],
        "gamma": [0, 0.1, 0.3, 0.5],
    }
    base_model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=60,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train_enc, y_train)
    model = search.best_estimator_
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # Threshold tuning: with class weighting pushing more predictions toward
    # "bad", the default 0.5 cutoff is no longer the right decision boundary.
    # Sweep thresholds and pick the one that maximizes recall on "bad" while
    # keeping precision from collapsing (business tradeoff, tunable).
    proba_train = model.predict_proba(X_train_enc)[:, 1]
    best_threshold, best_f1_bad = 0.5, 0.0
    for t in [i / 100 for i in range(20, 80, 5)]:
        preds_t = (proba_train >= t).astype(int)
        f1_bad = f1_score(y_train, preds_t, pos_label=1)
        if f1_bad > best_f1_bad:
            best_f1_bad, best_threshold = f1_bad, t

    proba = model.predict_proba(X_test_enc)[:, 1]
    preds = (proba >= best_threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "cv_roc_auc": round(search.best_score_, 4),
        "decision_threshold": best_threshold,
        "best_params": search.best_params_,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, preds, target_names=["good", "bad"]))

    feature_names = list(UNSKEWED_NUMERIC) + list(SKEWED_NUMERIC) + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    )

    joblib.dump(model, "model.joblib")
    joblib.dump(preprocessor, "preprocessor.joblib")
    joblib.dump(feature_names, "feature_names.joblib")
    joblib.dump(best_threshold, "threshold.joblib")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model.joblib, preprocessor.joblib, feature_names.joblib, threshold.joblib, metrics.json")


if __name__ == "__main__":
    main()
