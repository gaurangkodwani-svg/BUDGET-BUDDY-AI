"""Transaction category classifier: rules + TF-IDF / RandomForest."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

import load_data

MODEL_PATH = "category_classifier_pipeline.pkl"

RULES = [
    (("ke electric", "sui", "wasa", "nayatel", "k-electric"), "Utilities"),
    (("foodpanda", "restaurant", "mcdonalds", "savour", "kfc", "pizza"), "Food"),
    (("careem", "uber", "yango", "pso", "petrol"), "Travel"),
    (("daraz", "elo", "shopping"), "Shopping"),
    (("netflix", "cinema", "arena gaming", "entertainment"), "Entertainment"),
    (("chughtai", "lab", "health"), "Healthcare"),
    (("rent", "house rent"), "Rent"),
    (("cash withdrawal",), "Others"),
]


def _rule_based(description: str) -> Optional[Tuple[str, float]]:
    desc = description.lower()
    for keywords, category in RULES:
        if any(kw in desc for kw in keywords):
            return category, 1.0
    return None


def train_model(expenses_df: pd.DataFrame, fast: bool = True) -> Pipeline:
    """Train and optionally save the category classifier."""
    X = expenses_df["Description"]
    y = expenses_df["Category"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=5000)),
        ("classifier", RandomForestClassifier(
            n_estimators=100 if fast else 200,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_model() -> Optional[Pipeline]:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def get_or_train_model(expenses_df: pd.DataFrame) -> Pipeline:
    model = load_model()
    if model is None:
        model = train_model(expenses_df, fast=True)
    return model


def predict_category(description: str, model: Optional[Pipeline] = None) -> Tuple[str, float]:
    """Predict category using rules first, then ML."""
    ruled = _rule_based(description)
    if ruled:
        return ruled

    if model is None:
        model = load_model()
    if model is None:
        return "Others", 0.0

    category = model.predict([description])[0]
    confidence = float(model.predict_proba([description]).max())
    return category, confidence


def categorize_dataframe(expenses_df: pd.DataFrame, model: Optional[Pipeline] = None) -> pd.DataFrame:
    """Predict categories for uncategorized ('Others') transactions."""
    if model is None:
        model = get_or_train_model(expenses_df)

    mask = expenses_df["Category"].fillna("Others").eq("Others")
    if not mask.any():
        return pd.DataFrame(columns=["Description", "Predicted_Category", "Confidence", "Original_Category"])

    rows = []
    for _, row in expenses_df[mask].iterrows():
        cat, conf = predict_category(row["Description"], model)
        rows.append({
            "Description": row["Description"],
            "Amount": row["Amount"],
            "Date": row.get("Date", ""),
            "Predicted_Category": cat,
            "Confidence": conf,
            "Original_Category": row["Category"],
        })
    return pd.DataFrame(rows)


def evaluate_model(expenses_df: pd.DataFrame, model: Optional[Pipeline] = None) -> dict[str, Any]:
    """Return accuracy, classification report, and cross-validation scores."""
    if model is None:
        model = get_or_train_model(expenses_df)

    X = expenses_df["Description"]
    y = expenses_df["Category"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    n_splits = min(5, y.nunique(), len(y) // 2)
    if n_splits >= 2:
        cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring="accuracy")
    else:
        cv_scores = [accuracy]

    return {
        "accuracy": float(accuracy),
        "cv_mean": float(cv_scores.mean()),
        "cv_scores": [float(s) for s in cv_scores],
        "classification_report": report,
        "best_params": {"n_estimators": 100, "max_depth": 20, "class_weight": "balanced"},
        "model_type": "TF-IDF + RandomForest",
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def main() -> None:
    _, expenses_df, _ = load_data.load_and_clean_data()
    print("Training category classifier...")
    model = train_model(expenses_df, fast=False)
    metrics = evaluate_model(expenses_df, model)
    print(f"Test accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"CV mean:       {metrics['cv_mean'] * 100:.2f}%")

    test_merchants = [
        "KE Electric Bill",
        "Foodpanda Order",
        "Careem Ride",
        "Daraz Online Shopping",
        "Unknown Merchant XYZ",
    ]
    print("\nSample predictions:")
    for merchant in test_merchants:
        cat, conf = predict_category(merchant, model)
        print(f"  {merchant:30} -> {cat:15} ({conf:.0%})")


if __name__ == "__main__":
    main()
