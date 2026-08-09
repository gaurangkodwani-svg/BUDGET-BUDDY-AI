"""Load, validate, and clean bank statement CSV data."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import pandas as pd

REQUIRED_COLUMNS = {"Date", "Description", "Amount", "Type"}
OPTIONAL_COLUMNS = {"Category"}
DEFAULT_CSV = "pakistan_statement.csv"


def validate_csv(df: pd.DataFrame) -> dict:
    """Return a validation report for an uploaded or loaded statement."""
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "row_count": len(df),
        "columns_found": list(df.columns),
    }

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        report["valid"] = False
        report["errors"].append(f"Missing required columns: {', '.join(sorted(missing))}")

    if "Amount" in df.columns and df["Amount"].isna().any():
        report["warnings"].append(f"{df['Amount'].isna().sum()} rows have missing amounts.")

    if "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        bad_dates = parsed.isna().sum()
        if bad_dates:
            report["warnings"].append(f"{bad_dates} rows have unparseable dates.")

    if "Type" in df.columns:
        known = {"Credit", "Debit", "credit", "debit", "income", "expense"}
        unknown = set(df["Type"].dropna().unique()) - known
        if unknown:
            report["warnings"].append(f"Unusual Type values: {', '.join(map(str, unknown))}")

    if "Category" in df.columns:
        others = (df["Category"].fillna("Others") == "Others").sum()
        if others:
            report["warnings"].append(f"{others} transactions marked as 'Others' or uncategorized.")

    return report


def _read_csv(source: Union[str, Path, BinaryIO]) -> pd.DataFrame:
    if hasattr(source, "read"):
        source.seek(0)
        return pd.read_csv(source)
    return pd.read_csv(source)


def load_and_clean_data(
    source: Union[str, Path, BinaryIO] = DEFAULT_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a bank statement CSV and return full, expense, and income DataFrames.

    Accepts a file path or a file-like object (e.g. Streamlit upload).
    """
    df = _read_csv(source)
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.strftime("%Y-%m")

    if "Category" not in df.columns:
        df["Category"] = "Others"
    df["Category"] = df["Category"].fillna("Others")

    if "Type" in df.columns:
        type_norm = df["Type"].astype(str).str.strip().str.lower()
        credit_mask = type_norm.isin(["credit", "income", "deposit"])
        debit_mask = type_norm.isin(["debit", "expense", "withdrawal"])
        income_df = df[credit_mask].copy()
        expenses_df = df[debit_mask].copy()
    else:
        income_df = df[df["Amount"] > 0].copy()
        expenses_df = df[df["Amount"] < 0].copy()
        expenses_df["Amount"] = expenses_df["Amount"].abs()

    return df, expenses_df, income_df


def print_month_summary(csv_path: str = DEFAULT_CSV) -> None:
    """CLI helper: print summary for the earliest month in the dataset."""
    df, _, _ = load_and_clean_data(csv_path)
    target_month = df["Month"].min()
    month_df = df[df["Month"] == target_month]
    month_credits = month_df[month_df["Type"].str.lower() == "credit"]
    month_debits = month_df[month_df["Type"].str.lower() == "debit"]

    print("=" * 42)
    print(f"FINANCIAL STATEMENT FOR THE MONTH: {target_month}")
    print("=" * 42)
    print(f"\nTotal credit transactions: {len(month_credits)}")
    print(f"Total debit transactions:  {len(month_debits)}")
    print("\nSample debits:")
    print(month_debits[["Date", "Description", "Amount", "Type"]].head(15).to_string(index=False))


def main() -> None:
    print_month_summary()


if __name__ == "__main__":
    main()
