# Customer Lifetime Value Prediction Engine

This project estimates each customer's revenue over the next six months and combines that estimate with short-term inactivity risk to suggest retention priorities. It includes a reproducible data and modeling pipeline, a Streamlit dashboard, and a batch prediction function.

## Run it on Windows

The easiest way is to double-click **`run_project.bat`**. Or open PowerShell in this folder and run:

```powershell
.\run_project.ps1
```

The launcher creates a local `.venv`, installs the pinned requirements, builds the reports and predictions if they do not exist yet, and starts the dashboard. Open the local URL printed by Streamlit (usually `http://localhost:8501`). Press **Ctrl+C** in the terminal to stop it.

To rebuild the pipeline outputs and then start the dashboard:

```powershell
.\run_project.ps1 -Rebuild
```

### Requirements and data

- Windows with Python **3.11** installed. The launcher can use the `py` launcher (`py -3.11`) or a `python` command that points to Python 3.11.
- The UCI Online Retail II workbook at `data/raw/online_retail_II.xlsx` for a first pipeline run. It is already present in this local workspace, but raw data is excluded from Git. If you clone the project elsewhere, download it from the [UCI dataset page](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place the workbook at that path. The dataset is licensed CC BY 4.0; cite Chen (2012), [Online Retail II](https://doi.org/10.24432/C5CG6D).

The launcher reuses `data/processed/customer_predictions.csv` when it exists. Use `-Rebuild` after changing the source data or pipeline. To open the dashboard without the launcher, first install the requirements, then run `python -m streamlit run app/streamlit_app.py` from the project folder.

## What happens in the pipeline

```text
UCI workbook (or transaction CSV)
        ↓
Clean and audit transactions             src/data_prep.py
        ↓
Build cutoff-safe features and targets   src/features.py
        ↓
Compare models and score customers       src/models.py, src/predict.py
        ↓
Write reports, figures, and predictions  run_pipeline.py
        ↓
Explore results or upload transactions   app/streamlit_app.py
```

Features use transactions before 1 March 2011; the revenue target covers March through August 2011. The models compared are a historical-average baseline, BG/NBD + Gamma-Gamma, and gradient boosting. A random forest estimates whether a customer makes no purchase in the following 90 days. Customer segments combine predicted value and inactivity risk.

## Project map

| Path | Purpose |
| --- | --- |
| `run_project.bat`, `run_project.ps1` | Windows one-command setup and dashboard launcher; `-Rebuild` regenerates outputs |
| `run_pipeline.py` | Runs cleaning, analysis, model evaluation, customer scoring, and report generation |
| `src/data_prep.py` | Loads Excel or CSV transactions, cleans them, and writes a cleaning audit |
| `src/features.py` | Builds customer features using only data before each cutoff |
| `src/models.py`, `src/predict.py` | Fits/evaluates models and exposes the batch prediction function |
| `src/evaluate.py` | Calculates lift and creates model evaluation charts |
| `app/` | Modular Streamlit + Plotly dashboard (`streamlit_app.py`, `charts.py`, `components.py`, `insights.py`) for portfolio overview, customer segments, customer lookup, model performance, historical sales activity explorer, batch file scoring, and glossary |
| `notebooks/` | Guided exploratory analysis, feature engineering, and modeling notebooks |
| `data/raw/` | Source workbook or input transactions (ignored by Git) |
| `data/processed/` | Cleaned data, audit, and customer predictions (ignored by Git) |
| `reports/` | Executive summary, model metrics, segment/action tables, scenario, and figures |
| `models/` | Saved model bundle (ignored by Git) |

## Dashboard pages

Run `python -m streamlit run app/streamlit_app.py` (or use `run_project.bat`) to explore:

- **Portfolio overview:** KPI summary tiles, auto-generated key takeaways, zero-baseline segment revenue bar chart, segment treemap, Pareto/Lorenz concentration curve, log-scale CLV vs. churn scatter, and a downloadable high-risk retention priority list.
- **Customer segments:** Profile cards with recommended actions, CLV/risk distributions, and filtered segment export.
- **Customer lookup:** Customer ID autocomplete with portfolio rank/percentile, historical order timeline, top products, and country.
- **Model performance:** Highlighted model comparison table, plain-English verdict, cumulative gains, decile lift, and permutation feature importance.
- **Sales activity explorer:** Historical transaction explorer (Dec 2009–Dec 2011, clearly separated from forecasts) with Day/Week/Month granularity, period-over-period and YoY deltas, Q4 seasonality and anomaly highlights, weekday-by-hour heatmap, single day/week drill-down, and CSV export.
- **Score a file & How to read this:** Interactive batch scoring via `src.predict.predict_file` and a metric glossary.

## Run the pieces manually

From PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_pipeline.py
python -m streamlit run app/streamlit_app.py
```

If `py -3.11` is unavailable, install Python 3.11 and use its `python` command to create the virtual environment. To clean data without running model training, use `python -m src.data_prep`.

For your own transaction file, the Python API is `src.predict.predict_file(transaction_file, cutoff, output_file=None, model_path=None)`. It returns and writes customer ID, predicted six-month CLV, inactivity probability, and segment. Input columns should follow Online Retail II naming (`Invoice`, `StockCode`, `InvoiceDate`, `Quantity`, `Price`, `Customer ID`); the loader also accepts `InvoiceNo`, `UnitPrice`, and `CustomerID` aliases.

## Outputs

- `data/processed/customer_predictions.csv` — customer-level forecast, risk, and segment used by the dashboard.
- `data/processed/cleaning_audit.csv` — row counts removed at each cleaning step.
- `reports/model_comparison.csv`, `reports/decile_lift.csv`, `reports/feature_importance.csv` — evaluation results.
- `reports/segment_summary.csv`, `reports/segment_actions.csv`, `reports/roi_scenario.csv` — segment summaries and an illustrative planning scenario.
- `reports/figures/` — revenue, concentration, product/country, frequency, and model-gains charts.
- `reports/executive_summary.md` — findings and recommended actions for a quick overview.

## Results and limitations

On the chronological March–August 2011 holdout, BG/NBD + Gamma-Gamma achieved £558.32 MAE, £1,661.35 RMSE, and 0.782 R². Its highest predicted quintile contained 77.8% of future revenue. The tree model performed worse on this holdout. See `reports/model_comparison.csv` for the full comparison.

These are historical results from one UK retailer, with data ending in 2011. Transaction revenue is not profit. The ROI scenario assumes a 30% contribution margin, 10% incremental retained revenue, and £1 per contact; it is illustrative, not a measured campaign outcome. Validate on current data and use a controlled campaign test before making budget decisions.

## Troubleshooting

- **Python 3.11 not found:** install Python 3.11 and ensure the `py` launcher or the matching `python` command is available, then rerun the launcher.
- **Dataset missing:** put `online_retail_II.xlsx` in `data/raw/`, then run `./run_project.ps1 -Rebuild`.
- **Dashboard shows no portfolio data:** run `./run_project.ps1 -Rebuild` to regenerate the processed predictions.
- **Dashboard is already running:** stop the existing process with Ctrl+C before starting another one.
