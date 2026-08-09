"""Generate financial charts and save them to a dedicated output directory."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

CHART_NAMES = [
    "monthly_spending_trend.png",
    "category_donut_chart.png",
    "top_10_merchants_bar.png",
]


def generate_plots(
    df: pd.DataFrame,
    expenses_df: pd.DataFrame,
    output_dir: str | Path = "outputs/charts",
) -> List[str]:
    """Generate charts and return list of saved file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    if expenses_df.empty:
        return saved

    work = expenses_df.copy()
    if "Date" in work.columns:
        work["Month"] = work["Date"].dt.to_period("M").astype(str)

    # Chart 1: Monthly spending trend
    monthly_spending = work.groupby("Month")["Amount"].sum()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_spending.index, monthly_spending.values, marker="o", linewidth=2, color="#4f46e5")
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Spending (PKR)", fontsize=12)
    plt.title("Monthly Spending Trend", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = out / "monthly_spending_trend.png"
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close()
    saved.append(str(path1))

    # Chart 2: Category donut
    category_spending = work.groupby("Category")["Amount"].sum()
    plt.figure(figsize=(10, 10))
    plt.pie(
        category_spending.values,
        labels=category_spending.index,
        autopct="%1.1f%%",
        colors=plt.cm.Set3.colors,
        startangle=90,
        textprops={"fontsize": 11},
    )
    centre = plt.Circle((0, 0), 0.70, fc="white")
    plt.gca().add_artist(centre)
    plt.title("Category-wise Spending Distribution", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    path2 = out / "category_donut_chart.png"
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close()
    saved.append(str(path2))

    # Chart 3: Top 10 merchants
    top_merchants = work.groupby("Description")["Amount"].sum().nlargest(10)
    plt.figure(figsize=(10, 7))
    bars = plt.barh(top_merchants.index, top_merchants.values, color="skyblue", edgecolor="navy")
    plt.xlabel("Total Amount (PKR)", fontsize=12)
    plt.ylabel("Merchant", fontsize=12)
    plt.title("Top 10 Merchants by Spending", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"PKR {width:,.0f}",
            ha="left",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    path3 = out / "top_10_merchants_bar.png"
    plt.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close()
    saved.append(str(path3))

    return saved


def main() -> None:
    import load_data

    df, expenses_df, _ = load_data.load_and_clean_data()
    paths = generate_plots(df, expenses_df)
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
