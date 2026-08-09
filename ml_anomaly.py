"""Isolation Forest anomaly detection for expense transactions."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

import load_data

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _build_features(expenses_df: pd.DataFrame) -> pd.DataFrame:
    work = expenses_df.copy()
    work["DayOfWeek"] = work["Date"].dt.dayofweek
    encoder = LabelEncoder()
    work["Category_encoded"] = encoder.fit_transform(work["Category"].astype(str))
    return work


def _explain_anomaly(row: pd.Series, category_median: float, category_mean: float) -> str:
    amount = float(row["Amount"])
    day_name = DAY_NAMES[int(row["DayOfWeek"])]
    ratio = amount / category_median if category_median > 0 else 0.0

    parts = [
        f"Amount PKR {amount:,.0f} is {ratio:.1f}x the typical {row['Category']} spend "
        f"(median PKR {category_median:,.0f}).",
        f"Occurred on a {day_name}.",
        f"Anomaly score: {row['Anomaly_Score']:.3f} (lower = more unusual).",
    ]
    if amount > category_mean * 2:
        parts.append("Significantly above category average.")
    return " ".join(parts)


def detect_anomalies(expenses_df: pd.DataFrame, contamination: float = 0.05) -> list[dict[str, Any]]:
    """
    Detect anomalous debit transactions using Isolation Forest.

    Returns a list of dicts including human-readable explanations.
    """
    if expenses_df is None or len(expenses_df) < 5:
        return []

    work = _build_features(expenses_df)
    features = work[["Amount", "DayOfWeek", "Category_encoded"]]

    model = IsolationForest(
        n_estimators=100,
        contamination=min(contamination, 0.49),
        random_state=42,
    )
    model.fit(features)
    work["Anomaly"] = model.predict(features)
    work["Anomaly_Score"] = model.decision_function(features)

    category_stats = work.groupby("Category")["Amount"].agg(["median", "mean"])

    anomalous = work[work["Anomaly"] == -1].sort_values("Anomaly_Score")
    results = []
    for _, row in anomalous.iterrows():
        stats = category_stats.loc[row["Category"]] if row["Category"] in category_stats.index else pd.Series(
            {"median": work["Amount"].median(), "mean": work["Amount"].mean()}
        )
        results.append(
            {
                "Date": row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "",
                "Description": row["Description"],
                "Amount": float(row["Amount"]),
                "Category": row["Category"],
                "Anomaly_Score": float(row["Anomaly_Score"]),
                "Explanation": _explain_anomaly(row, float(stats["median"]), float(stats["mean"])),
            }
        )
    return results


def get_anomaly_metrics(expenses_df: pd.DataFrame) -> dict[str, Any]:
    """Return model statistics for the metrics dashboard."""
    if expenses_df is None or len(expenses_df) < 5:
        return {"trained": False, "message": "Not enough transactions for anomaly detection."}

    work = _build_features(expenses_df)
    features = work[["Amount", "DayOfWeek", "Category_encoded"]]
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(features)
    work["Anomaly"] = model.predict(features)
    work["Anomaly_Score"] = model.decision_function(features)

    normal = work[work["Anomaly"] == 1]
    anomalous = work[work["Anomaly"] == -1]

    return {
        "trained": True,
        "model": "Isolation Forest",
        "n_estimators": 100,
        "contamination": 0.05,
        "total_transactions": len(work),
        "anomaly_count": len(anomalous),
        "normal_avg_amount": float(normal["Amount"].mean()) if len(normal) else 0.0,
        "anomaly_avg_amount": float(anomalous["Amount"].mean()) if len(anomalous) else 0.0,
        "anomaly_by_category": anomalous["Category"].value_counts().to_dict() if len(anomalous) else {},
    }


def main() -> None:
    _, expenses_df, _ = load_data.load_and_clean_data()
    anomalies = detect_anomalies(expenses_df)
    print(f"Anomalies found: {len(anomalies)}")
    for item in anomalies[:10]:
        print(f"\n{item['Date']} | {item['Description']} | PKR {item['Amount']:,.0f}")
        print(f"  {item['Explanation']}")


if __name__ == "__main__":
    main()
