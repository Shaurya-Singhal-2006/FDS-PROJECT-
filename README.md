# Customer Lifetime Value Prediction Engine

An end-to-end, explainable project that ranks customers by expected next-six-month revenue and pairs value with inactivity risk to guide retention actions.

## Dataset and provenance

Uses the real [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii), downloaded from the UCI static dataset endpoint. UCI describes 1,067,371 transaction lines from a UK non-store retailer between 1 December 2009 and 9 December 2011. License: CC BY 4.0. Cite Chen (2012), [Online Retail II](https://doi.org/10.24432/C5CG6D). The workbook is kept locally in `data/raw/` and excluded from Git. No synthetic records are mixed in.

## Approach

1. Audit exclusions and clean purchase transactions; retain the 18,744 customer-linked negative-quantity rows separately for a historical return-rate feature. All of those rows have C-prefixed invoices and are excluded from purchase modeling with cancellations.
2. Explore monthly revenue, country/product concentration, order frequency, repeat purchasing and Pareto concentration.
3. Build features strictly before 1 March 2011 and label them with actual revenue from 1 March through 31 August 2011.
4. Compare historical-average, BG/NBD + Gamma-Gamma, and gradient boosting regression, plus a random-forest classifier for no purchase in the first 90 days after cutoff. Gradient-boosting regularization is selected on the earlier September 2010 snapshot; the final evaluation uses the March 2011 cutoff and actual March–August 2011 revenue. Every training target window ends before that final holdout begins.
5. Assign Champions, Loyal, Potential, At-Risk and Hibernating segments from forecast value and risk.

Feature cutoff prevents leakage: no transaction on or after a snapshot cutoff enters that snapshot's RFM, tenure, product, interval, Q4 or return features. Six-month targets are constructed separately from future rows. The 90-day inactivity label is computed only from future rows and is never included as a feature.

## Results

Measured on the chronological validation snapshot (6-month future revenue, GBP):

| Model | MAE | RMSE | R² | Future revenue captured by top 20% |
|---|---:|---:|---:|---:|
| Historical average | £573.32 | £2,004.53 | 0.6832 | 76.5% |
| BG/NBD + Gamma-Gamma | **£558.32** | **£1,661.35** | **0.7824** | **77.8%** |
| Gradient boosting | £1,241.12 | £3,017.85 | 0.2818 | 76.4% |

BG/NBD + Gamma-Gamma has the lowest holdout error and captures the most revenue in its top quintile. Its top predicted decile averages 6.39× the validation-wide mean revenue. Gradient boosting underperforms both alternatives on this time holdout. Monetary history is the strongest positive permutation-importance driver in the gradient-boosting analysis.

There are 5,878 identified customers across the cleaned data period; 4,537 had purchase history before the 1 March 2011 scoring cutoff and received a forecast. Champions number 887 (71.5% of predicted revenue); At-Risk has 21 customers (2.0%); Loyal 1,192 (17.4%); Potential 1,672 (6.8%); Hibernating 765 (2.4%). Full values and actions are in `reports/segment_actions.csv`.

The optional planning scenario selects Champions + At-Risk (908 customers). It assumes 30% contribution margin, 10% incremental retained revenue, and £1 per contact. Under those assumptions the estimated incremental profit is £93,229, campaign cost £908, and net benefit £92,321 (10,168% ROI). This is an illustrative scenario, not a measured campaign outcome or causal forecast; replace assumptions with experiment results before budgeting.

## Quick start

Requires Python 3.11 and the UCI workbook at `data/raw/online_retail_II.xlsx`.

**Windows:** double-click `run_project.bat`, or run `./run_project.ps1` from PowerShell. The launcher creates `.venv`, installs the pinned packages, builds the outputs on the first run, and starts the Streamlit dashboard. Later runs reuse the existing predictions. Run `./run_project.ps1 -Rebuild` to regenerate the pipeline outputs.

If the dataset workbook is missing, download Online Retail II from the [UCI page](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place `online_retail_II.xlsx` in `data/raw/`.

### Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_pipeline.py
streamlit run app/streamlit_app.py
pytest -q
```

Or run cleaning alone with `python -m src.data_prep`. The prediction API is `src.predict.predict_file(transaction_file, cutoff, output_file=None, model_path=None)` and writes customer ID, non-negative predicted six-month CLV, churn probability and segment. The joblib bundle in `models/` stores the BG/NBD + Gamma-Gamma calibration data and refits the Lifetimes components when loaded because Lifetimes fitter objects contain callbacks that joblib cannot serialize directly.

## Visuals

- Monthly revenue: `reports/figures/monthly_revenue.png`
- Country/product leaders: `reports/figures/top_countries.png`, `top_products.png`
- Frequency and Pareto: `order_frequency.png`, `pareto_curve.png`
- Cumulative gains: `cumulative_gains.png`
- Dashboard screenshot: add a screenshot at `reports/figures/dashboard_screenshot.png` after launching Streamlit.

## Limitations and future work

Transaction-only features omit customer demographics, channel exposure and campaign response. This is one UK retailer and historical data ends in 2011. Revenue is not contribution margin. The six-month target horizon and inactivity definition are operational choices. Validate on recent data, add campaign cost/margin and controlled experiments, monitor drift, and calibrate probability estimates before production use.

## Resume bullets

- Built a leakage-safe six-month CLV pipeline in Python on 1.07M UCI transaction rows; compared three models chronologically and achieved £558.32 MAE and 0.782 R² with BG/NBD + Gamma-Gamma.
- Engineered ten RFM, tenure, order-interval, assortment, return and seasonal features; the leading predicted decile averaged 6.39× holdout-wide revenue.
- Developed a Streamlit dashboard for five value/risk segments and batch scoring; identified a 908-customer Champions + At-Risk pilot cohort for an explicitly assumption-based ROI scenario.

## Skills

Python, pandas, scikit-learn, lifetimes, feature engineering, time-aware validation, customer analytics, CLV modeling, classification, model evaluation, data visualization, Streamlit, reproducible workflows.

## LinkedIn draft

I built a Customer Lifetime Value Prediction Engine using the UCI Online Retail II dataset. It cleans and audits transaction data, creates leakage-safe customer features, compares historical-average, BG/NBD + Gamma-Gamma and gradient-boosting approaches, and turns value plus inactivity risk into five actionable segments. On a chronological six-month holdout, BG/NBD + Gamma-Gamma achieved £558.32 MAE and 0.782 R²; the top predicted decile averaged 6.39× the overall customer revenue. I also built a Streamlit dashboard for exploration and batch scoring. Transaction revenue is not profit, the data is historical and UK-specific, and ROI estimates need campaign evidence. #DataScience #CustomerAnalytics #Python

