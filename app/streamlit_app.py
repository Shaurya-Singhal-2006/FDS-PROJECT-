"""Streamlit dashboard for customer value forecasts and retention priorities."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import predict_file


st.set_page_config(
    page_title="Customer Value | Retention Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f4f7fb; color: #182230; }
    [data-testid="stAppViewContainer"] { background: #f4f7fb; }
    [data-testid="stHeader"] { background: rgba(255,255,255,.92); }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e4eaf2;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label { color: #344054; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3, p, label { color: #182230; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4eaf2;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(16,24,40,.04);
    }
    [data-testid="stMetricLabel"] { color: #667085; }
    [data-testid="stMetricValue"] { color: #182230; }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e4eaf2;
        border-radius: 10px;
        overflow: hidden;
    }
    .eyebrow { color: #315fbd; font-size: .78rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase; margin-bottom: .25rem; }
    .subtle { color: #667085; }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Portfolio overview",
    "Customer segments",
    "Customer lookup",
    "Model performance",
    "Score a file",
    "How to read this",
]
SEGMENT_ACTIONS = {
    "Champions": "Protect with VIP service; avoid routine discounts.",
    "Loyal": "Recommend relevant products and replenishment.",
    "Potential": "Encourage a second purchase with a low-cost nudge.",
    "At-Risk": "Test a personalized win-back offer.",
    "Hibernating": "Try capped reactivation and suppress non-responders.",
}
SEGMENT_ORDER = ["Champions", "Loyal", "Potential", "At-Risk", "Hibernating"]


def load_predictions():
    """Load pipeline output, or return an empty table before the first run."""
    path = ROOT / "data" / "processed" / "customer_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def page_heading(title, description):
    st.markdown('<div class="eyebrow">Retention analytics</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def require_predictions(data):
    if not data.empty:
        return True
    st.info(
        "No customer forecast file was found. Run `run_project.bat` or "
        "`python run_pipeline.py` from the project folder to create it."
    )
    return False


def customer_table(data):
    """Give customer data human-readable labels and display formats."""
    renamed = data.rename(
        columns={
            "Customer ID": "Customer ID",
            "predicted_6m_clv": "Forecast value (6 months)",
            "churn_probability": "Inactivity risk",
            "segment": "Priority group",
        }
    )
    renamed["Customer ID"] = renamed["Customer ID"].astype(str)
    return renamed


def show_customer_table(data):
    st.dataframe(
        customer_table(data),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Customer ID": st.column_config.TextColumn("Customer ID"),
            "Forecast value (6 months)": st.column_config.NumberColumn(
                "Forecast value (6 months)", format="£%.2f"
            ),
            "Inactivity risk": st.column_config.NumberColumn(
                "Inactivity risk", format="percent"
            ),
            "Priority group": st.column_config.TextColumn("Priority group"),
        },
    )


def portfolio_overview(data):
    page_heading(
        "Customer value portfolio",
        "A six-month revenue forecast paired with a 90-day inactivity risk estimate.",
    )
    if not require_predictions(data):
        return

    total_value = data["predicted_6m_clv"].sum()
    high_risk = int(data["churn_probability"].ge(0.5).sum())
    a, b, c, d = st.columns(4)
    a.metric("Scored customers", f"{len(data):,}")
    b.metric("Forecast portfolio revenue", f"£{total_value:,.0f}")
    c.metric("Higher inactivity risk", f"{high_risk:,}")
    d.metric("Priority groups", f"{data['segment'].nunique()}")

    st.subheader("Forecast value by customer group")
    st.caption(
        "Each bar sums the six-month forecast for customers in that group. "
        "This is forecast transaction revenue, not profit."
    )
    values = (
        data.groupby("segment")["predicted_6m_clv"]
        .sum()
        .reindex(SEGMENT_ORDER)
        .dropna()
    )
    st.bar_chart(values, color="#315fbd", height=320)

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Forecast value:** estimated revenue from each customer over the next six months.\n"
            "- **Inactivity risk:** estimated chance of no purchase in the first 90 days after the scoring cutoff.\n"
            "- **Higher inactivity risk:** customers with an estimated risk of at least 50%. This is a prioritization threshold, not a guarantee."
        )


def customer_segments(data):
    page_heading(
        "Customer segments",
        "See how forecast value and inactivity risk translate into suggested retention priorities.",
    )
    if not require_predictions(data):
        return

    grouped = data.groupby("segment").agg(
        customers=("Customer ID", "count"),
        forecast_value=("predicted_6m_clv", "sum"),
        average_risk=("churn_probability", "mean"),
    )
    grouped["portfolio_share"] = grouped["forecast_value"] / grouped["forecast_value"].sum()
    grouped["suggested_next_step"] = [SEGMENT_ACTIONS.get(name, "") for name in grouped.index]
    grouped = grouped.reindex([s for s in SEGMENT_ORDER if s in grouped.index]).reset_index()
    grouped = grouped.rename(
        columns={
            "segment": "Customer group",
            "customers": "Customers",
            "forecast_value": "Forecast value",
            "average_risk": "Average inactivity risk",
            "portfolio_share": "Share of forecast value",
            "suggested_next_step": "Suggested next step",
        }
    )
    st.dataframe(
        grouped,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Customers": st.column_config.NumberColumn("Customers", format="%d"),
            "Forecast value": st.column_config.NumberColumn("Forecast value", format="£%.0f"),
            "Average inactivity risk": st.column_config.NumberColumn(
                "Average inactivity risk", format="percent"
            ),
            "Share of forecast value": st.column_config.NumberColumn(
                "Share of forecast value", format="percent"
            ),
            "Suggested next step": st.column_config.TextColumn("Suggested next step", width="large"),
        },
    )

    st.subheader("Inspect customers")
    chosen = st.selectbox("Choose a customer group", SEGMENT_ORDER, key="segment_filter")
    selected = data[data["segment"] == chosen].sort_values(
        "predicted_6m_clv", ascending=False
    )
    st.caption(f"{len(selected):,} customers · sorted by forecast value, highest first")
    show_customer_table(selected)


def customer_lookup(data):
    page_heading(
        "Customer lookup",
        "Search the scored customer list by ID and inspect its forecast and risk estimate.",
    )
    if not require_predictions(data):
        return
    query = st.text_input("Customer ID", placeholder="Enter all or part of an ID")
    if not query.strip():
        st.caption(f"{len(data):,} customers are available. Enter an ID to search.")
        return
    matches = data[
        data["Customer ID"].astype(str).str.contains(query.strip(), case=False, regex=False)
    ].sort_values("predicted_6m_clv", ascending=False)
    st.caption(f"{len(matches):,} matching customer(s)")
    if matches.empty:
        st.warning("No customer IDs matched that search.")
    else:
        show_customer_table(matches)


def model_performance():
    page_heading(
        "Model performance",
        "A time-based holdout compares forecast accuracy and how well models rank future revenue.",
    )
    metrics_path = ROOT / "reports" / "model_comparison.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).rename(
            columns={
                "model": "Model",
                "MAE": "Mean absolute error",
                "RMSE": "Root mean squared error",
                "R2": "R²",
                "Top 20% revenue captured": "Revenue in top predicted 20%",
            }
        )
        st.dataframe(
            metrics,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Model": st.column_config.TextColumn("Model"),
                "Mean absolute error": st.column_config.NumberColumn(
                    "Mean absolute error", help="Average absolute error in GBP.", format="£%.2f"
                ),
                "Root mean squared error": st.column_config.NumberColumn(
                    "Root mean squared error", help="Penalizes larger errors more heavily.", format="£%.2f"
                ),
                "R²": st.column_config.NumberColumn(
                    "R²", help="How much variation in holdout revenue the model explains.", format="%.3f"
                ),
                "Revenue in top predicted 20%": st.column_config.NumberColumn(
                    "Revenue in top predicted 20%",
                    help="Share of actual holdout revenue found among the top fifth of customers ranked by forecast.",
                    format="percent",
                ),
            },
        )
    else:
        st.info("No model comparison is available yet. Run the full pipeline to generate it.")

    gains_path = ROOT / "reports" / "figures" / "cumulative_gains.png"
    if gains_path.exists():
        st.subheader("Revenue captured as the target list grows")
        st.caption(
            "The curve ranks customers by forecast value. A steeper curve means a smaller "
            "target group captures more of the holdout revenue."
        )
        st.image(str(gains_path), use_container_width=True)

    with st.expander("How to interpret this evaluation"):
        st.markdown(
            "MAE and RMSE measure forecast error in pounds; lower is better. R² measures "
            "fit on the holdout; higher is better. Revenue captured in the top predicted "
            "20% measures ranking usefulness. These are historical results from a single "
            "retailer, not a guarantee of future performance."
        )


def score_file():
    page_heading(
        "Score a transaction file",
        "Upload a transaction history to generate six-month customer forecasts and retention groups.",
    )
    st.info(
        "The file must contain Online Retail II-style invoice, item, date, quantity, "
        "price, and customer ID columns. Scoring fits a model from the uploaded history."
    )
    uploaded = st.file_uploader("Transaction file", type=["csv", "xlsx"])
    cutoff = st.date_input(
        "Feature cutoff date",
        value=pd.Timestamp("2011-03-01").date(),
        help="Only transactions before this date are used as customer features.",
    )
    if uploaded and st.button("Generate forecasts", type="primary"):
        data_dir = ROOT / "data" / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(uploaded.name).suffix.lower()
        temp_path = data_dir / f"upload_for_scoring{suffix}"
        try:
            temp_path.write_bytes(uploaded.getvalue())
            with st.spinner("Cleaning transactions and fitting the scoring models…"):
                result = predict_file(temp_path, cutoff)
            st.success(f"Generated forecasts for {len(result):,} customers.")
            show_customer_table(result)
            st.download_button(
                "Download customer forecasts",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="customer_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")
        finally:
            temp_path.unlink(missing_ok=True)


def how_to_read():
    page_heading(
        "How to read this dashboard",
        "A short guide to the forecast, the risk estimate, the groups, and the workflow behind them.",
    )
    st.subheader("What happens to the data")
    st.markdown(
        "1. **Clean:** remove unusable rows, cancellations, and duplicate transactions; keep an audit of the removals.\n"
        "2. **Build customer history:** calculate order frequency, spending, recency, product breadth, and other features before a chosen cutoff.\n"
        "3. **Forecast:** estimate revenue during the following six months and the chance of no purchase during the first 90 days.\n"
        "4. **Prioritize:** combine forecast value and inactivity risk into five customer groups for review."
    )

    st.subheader("Definitions")
    st.markdown(
        "- **Six-month forecast value:** estimated transaction revenue for one customer in the six months after the cutoff. It is not profit or customer lifetime value over the entire relationship.\n"
        "- **Inactivity risk:** model-estimated probability of no purchase in the first 90 days after the cutoff.\n"
        "- **Customer group:** a rules-based label that combines relative forecast value and inactivity risk. It suggests where to review outreach; it does not prescribe a campaign.\n"
        "- **Holdout:** a later period kept out of model fitting, used to compare predictions with actual revenue."
    )

    st.subheader("How the groups are assigned")
    st.markdown(
        "**Champions** are among the highest-value, lower-risk customers. **Loyal** customers "
        "have above-median value and lower risk. **At-Risk** customers have high value and "
        "higher risk. **Hibernating** customers have very high inactivity risk. The remaining "
        "customers are **Potential**. Thresholds are relative to the scored portfolio."
    )

    st.warning(
        "This project uses historical transactions from one UK retailer, ending in 2011. "
        "Revenue is not margin. Treat forecasts and suggested actions as decision support, "
        "and validate any campaign with current data and a controlled test."
    )


st.sidebar.markdown("### Customer Value Engine")
st.sidebar.caption("Six-month forecasting · retention planning")
page = st.sidebar.radio("Dashboard", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Data source: UCI Online Retail II · historical data")

predictions = load_predictions()
if page == "Portfolio overview":
    portfolio_overview(predictions)
elif page == "Customer segments":
    customer_segments(predictions)
elif page == "Customer lookup":
    customer_lookup(predictions)
elif page == "Model performance":
    model_performance()
elif page == "Score a file":
    score_file()
else:
    how_to_read()
