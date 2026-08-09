# BudgetBuddy AI

Personal finance dashboard that analyzes bank statements with EDA, budget tracking, ML anomaly detection, auto-categorization, a what-if savings simulator, and a Groq-powered financial advisor.

## Features

- **CSV upload & validation** — Upload any bank statement or use the included sample data
- **Summary dashboard** — Income, expenses, net savings, category breakdown
- **Budget vs actual** — Set per-category budgets with 80% / 100% alerts
- **What-if simulator** — Model spending cuts and projected monthly savings
- **Visual analytics** — Monthly trend, category donut, top merchants charts
- **Anomaly detection** — Isolation Forest with human-readable explanations
- **Auto-categorization** — Rules + TF-IDF / RandomForest for Pakistani merchants
- **ML metrics** — Classifier accuracy, cross-validation, anomaly statistics
- **AI advisor** — Groq LLM with privacy mode (aggregates only)
- **Export** — Download HTML or PDF financial report

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Groq API key for AI Advisor

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run the app

```bash
streamlit run nlp.assistant.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## CSV format

| Column      | Required | Description                          |
|-------------|----------|--------------------------------------|
| Date        | Yes      | Transaction date (YYYY-MM-DD)        |
| Description | Yes      | Merchant or transaction description  |
| Amount      | Yes      | Transaction amount (positive number) |
| Type        | Yes      | `Credit` or `Debit`                  |
| Category    | No       | e.g. Food, Travel, Utilities         |

Sample file: `pakistan_statement.csv` (~1,000 transactions).

## Project structure

```
gaurang.ai/
├── nlp.assistant.py      # Streamlit app (main entry point)
├── load_data.py          # Load, validate, clean CSV
├── eda_numpy.py          # Exploratory data analysis
├── ml_anomaly.py         # Isolation Forest anomaly detection
├── ml_classifier.py      # Category classifier (rules + ML)
├── plot_matplotlib.py    # Chart generation
├── report_export.py      # HTML / PDF report export
├── pakistan_statement.csv
├── requirements.txt
└── outputs/charts/       # Generated charts (auto-created)
```

## Standalone scripts

Each module can also be run directly for CLI output:

```bash
python load_data.py
python eda_numpy.py
python ml_anomaly.py
python ml_classifier.py
python plot_matplotlib.py
```

Training the classifier saves `category_classifier_pipeline.pkl` for faster subsequent loads.

## Demo flow (for presentation)

1. Launch app → validation report shows row count
2. **Summary** — Review income, expenses, category chart
3. **Budget** — Adjust a category budget → show over-budget alert
4. **What-If** — Reduce Food spending 20% → show projected savings
5. **Anomalies** — Expand a flagged transaction → read explanation
6. **Auto-Categorize** — Predict category for a merchant name
7. **ML Metrics** — Show classifier accuracy and anomaly stats
8. **Export** — Download HTML report

## Tech stack

- **UI:** Streamlit
- **Data:** pandas, NumPy
- **ML:** scikit-learn (Isolation Forest, TF-IDF + RandomForest)
- **Charts:** matplotlib
- **LLM:** Groq (Llama 3.3 70B)
- **Export:** HTML, reportlab (PDF)

## Author

Gaurang — Software Engineering project
