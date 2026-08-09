"""Exploratory data analysis for bank statements."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import load_data


def perform_eda(
    df: pd.DataFrame,
    expenses_df: pd.DataFrame,
    income_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute financial summary metrics and category/month breakdowns."""
    total_income = float(income_df["Amount"].sum()) if len(income_df) else 0.0
    total_expenses = float(expenses_df["Amount"].sum()) if len(expenses_df) else 0.0
    net_savings = total_income - total_expenses

    if "Date" in df.columns and df["Date"].notna().any():
        days_count = max(df["Date"].dt.date.nunique(), 1)
    else:
        days_count = 30

    daily_average = total_expenses / days_count

    category_breakdown = (
        expenses_df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
        if len(expenses_df)
        else {}
    )

    monthly_spending: dict[str, float] = {}
    if len(expenses_df) and "Date" in expenses_df.columns:
        monthly = expenses_df.copy()
        monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
        monthly_spending = monthly.groupby("Month")["Amount"].sum().to_dict()

    top_merchants = (
        expenses_df.groupby("Description")["Amount"]
        .sum()
        .nlargest(10)
        .to_dict()
        if len(expenses_df)
        else {}
    )

    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "net_savings": net_savings,
        "daily_average": daily_average,
        "transaction_count": len(df),
        "expense_count": len(expenses_df),
        "income_count": len(income_df),
        "category_breakdown": category_breakdown,
        "monthly_spending": monthly_spending,
        "top_merchants": top_merchants,
        "unique_months": sorted(monthly_spending.keys()),
    }


def print_monthly_breakdown(csv_path: str = load_data.DEFAULT_CSV) -> None:
    """CLI helper: print month-by-month credit/debit samples."""
    df, expenses_df, income_df = load_data.load_and_clean_data(csv_path)

    for month in sorted(df["Month"].dropna().unique()):
        print(f"\n=== MONTH: {month} ===")
        m_credits = income_df[income_df["Month"] == month]
        m_debits = expenses_df[expenses_df["Month"] == month]

        if len(m_credits):
            row = m_credits.iloc[0]
            print(f"Credit: {row['Date'].date()} | {row['Description']} | +{row['Amount']} PKR")

        print("Debits:")
        for _, debit in m_debits.head(5).iterrows():
            print(f"  {debit['Date'].date()} | {debit['Description']} | -{debit['Amount']} PKR")


def main() -> None:
    summary = perform_eda(*load_data.load_and_clean_data())
    print("EDA Summary")
    print("-" * 40)
    for key, value in summary.items():
        if key not in {"category_breakdown", "monthly_spending", "top_merchants"}:
            print(f"{key}: {value}")
    print("\nCategory breakdown:")
    for cat, amt in summary["category_breakdown"].items():
        print(f"  {cat}: PKR {amt:,.2f}")


if __name__ == "__main__":
    main()
