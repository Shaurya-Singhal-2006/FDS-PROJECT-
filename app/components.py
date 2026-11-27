"""Reusable Streamlit UI components, styling helpers, and cached data loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.insights import (
    SEGMENT_COLORS,
    SEGMENT_ORDER,
    attach_customer_segments,
    filter_predictions,
    normalize_customer_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def inject_theme_css() -> None:
    """Inject theme-aware CSS that fixes title clipping and styles cards cleanly in light & dark mode."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1440px;
            padding-top: 3.2rem;
            padding-bottom: 3rem;
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stMetric"] {
            background: var(--secondary-background-color);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.80rem;
            font-weight: 600;
            opacity: 0.82;
        }
        [data-testid="stMetricLabel"] p {
            white-space: normal;
            line-height: 1.2;
        }
        [data-testid="stMetricValue"] {
            font-size: clamp(1.15rem, 1.45vw, 1.55rem);
            line-height: 1.25;
        }
        [data-testid="stMetricValue"] > div {
            white-space: normal;
            overflow-wrap: anywhere;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 12px;
            overflow: hidden;
        }
        .clv-hero {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(13, 148, 136, 0.08));
            border: 1px solid rgba(37, 99, 235, 0.24);
            border-radius: 16px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.25rem;
        }
        .clv-eyebrow {
            color: #2563EB;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .clv-hero-title {
            font-size: 1.95rem;
            font-weight: 750;
            line-height: 1.2;
            margin: 0 0 0.4rem 0;
            color: var(--text-color);
        }
        .clv-hero-subtitle {
            font-size: 0.96rem;
            opacity: 0.85;
            margin: 0;
        }
        .clv-headline-pill {
            margin-top: 0.85rem;
            padding: 0.65rem 0.9rem;
            border-radius: 10px;
            background: var(--secondary-background-color);
            border-left: 4px solid #2563EB;
            font-size: 0.93rem;
            font-weight: 500;
        }
        .clv-segment-card {
            background: var(--secondary-background-color);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            height: 100%;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }
        .clv-segment-badge {
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.55rem;
        }
        .clv-card-stat {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0.15rem 0;
        }
        .clv-card-meta {
            font-size: 0.82rem;
            opacity: 0.82;
            line-height: 1.45;
        }
        .clv-takeaway-box {
            background: var(--secondary-background-color);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            padding: 1rem 1.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    """Load cached customer-level 6-month forecasts."""
    path = ROOT / "data" / "processed" / "customer_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Customer ID" in df.columns:
        df["Customer ID"] = normalize_customer_ids(df["Customer ID"])
    return df


@st.cache_data(show_spinner="Loading cleaned historical transactions…")
def load_transactions_with_segments() -> pd.DataFrame:
    """Load cleaned historical transactions and join customer segment labels."""
    path = ROOT / "data" / "processed" / "transactions_clean.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(
        path,
        dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str, "Customer ID": str},
        parse_dates=["InvoiceDate"],
    )
    df["Customer ID"] = normalize_customer_ids(df["Customer ID"])
    preds = load_predictions()
    return attach_customer_segments(df, preds)


@st.cache_data(show_spinner=False)
def load_report_csv(filename: str) -> pd.DataFrame:
    """Load a generated CSV from reports/ or return empty DataFrame if missing."""
    path = ROOT / "reports" / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def render_hero_header(
    eyebrow: str,
    title: str,
    subtitle: str,
    headline_insight: str | None = None,
) -> None:
    """Render a spacious, non-clipped hero header with optional dynamic headline insight."""
    pill_html = (
        f'<div class="clv-headline-pill">💡 {headline_insight}</div>'
        if headline_insight
        else ""
    )
    st.markdown(
        f"""
        <div class="clv-hero">
            <div class="clv-eyebrow">{eyebrow}</div>
            <h1 class="clv-hero-title">{title}</h1>
            <p class="clv-hero-subtitle">{subtitle}</p>
            {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_predictions(df: pd.DataFrame) -> bool:
    """Show clear instructions when customer_predictions.csv is not generated yet."""
    if not df.empty:
        return True
    st.warning(
        "No customer forecast file (`data/processed/customer_predictions.csv`) was found. "
        "Place `online_retail_II.xlsx` in `data/raw/` and run `python run_pipeline.py` "
        "(or `run_project.bat`) from the project root to generate it."
    )
    return False


def require_transactions(df: pd.DataFrame) -> bool:
    """Show clear instructions when transactions_clean.csv is not generated yet."""
    if not df.empty:
        return True
    st.warning(
        "No cleaned transaction history (`data/processed/transactions_clean.csv`) was found. "
        "Place `online_retail_II.xlsx` in `data/raw/` and run `python run_pipeline.py` "
        "or `python -m src.data_prep` to generate it."
    )
    return False


def render_sidebar_portfolio_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render consistent global portfolio filters in the sidebar and return filtered rows."""
    if df.empty:
        return df

    st.sidebar.markdown("#### Portfolio filters")
    available_segments = [s for s in SEGMENT_ORDER if s in df["segment"].unique()]
    selected_segments = st.sidebar.multiselect(
        "Customer segments",
        options=available_segments,
        default=available_segments,
        help="Applies across Portfolio overview and Customer segments.",
    )

    churn_range = st.sidebar.slider(
        "90-day inactivity risk",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        format="%.2f",
    )

    max_clv = float(df["predicted_6m_clv"].max())
    slider_max = max(round(max_clv, 0), 1.0)
    clv_range = st.sidebar.slider(
        "Predicted 6-month CLV (£)",
        min_value=0.0,
        max_value=slider_max,
        value=(0.0, slider_max),
    )

    return filter_predictions(
        df,
        segments=selected_segments,
        churn_range=churn_range,
        clv_range=clv_range,
    )


def format_customer_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format customer prediction columns for clean table display."""
    out = df.copy()
    if "Customer ID" in out.columns:
        out["Customer ID"] = out["Customer ID"].astype(str)
    return out.rename(
        columns={
            "Customer ID": "Customer ID",
            "predicted_6m_clv": "Forecast value (6 months)",
            "churn_probability": "Inactivity risk",
            "segment": "Priority group",
        }
    )


def show_customer_table(df: pd.DataFrame, download_label: str | None = None, file_name: str = "customers.csv") -> None:
    """Display formatted customer predictions table with optional CSV download button."""
    if df.empty:
        st.info("No customers match the current selection.")
        return
    st.dataframe(
        format_customer_table(df),
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
    if download_label:
        st.download_button(
            download_label,
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=file_name,
            mime="text/csv",
        )


def render_segment_cards(segment_summary: pd.DataFrame) -> None:
    """Render one card per customer segment with revenue, risk, and recommended action."""
    if segment_summary.empty:
        st.info("No segments match the current filters.")
        return

    cols = st.columns(len(segment_summary))
    for col, (_, row) in zip(cols, segment_summary.iterrows()):
        seg = str(row["segment"])
        color = SEGMENT_COLORS.get(seg, "#2563EB")
        with col:
            st.markdown(
                f"""
                <div class="clv-segment-card" style="border-top: 4px solid {color};">
                    <div class="clv-segment-badge" style="background: {color};">{seg}</div>
                    <div class="clv-card-stat">£{row['predicted_revenue']:,.0f}</div>
                    <div class="clv-card-meta">
                        <b>{row['revenue_share']:.1%}</b> of forecast revenue<br>
                        <b>{int(row['customers']):,}</b> customers ({row['customer_share']:.1%})<br>
                        Avg CLV: <b>£{row['avg_clv']:,.0f}</b><br>
                        Avg risk: <b>{row['avg_churn']:.1%}</b>
                    </div>
                    <hr style="margin: 0.65rem 0; border-color: rgba(148,163,184,0.22);">
                    <div class="clv-card-meta">{row['recommended_action']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
