"""Interactive Streamlit dashboard for 6-month CLV forecasts and historical sales activity."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.charts import (
    PALETTE,
    clv_churn_scatter,
    cumulative_gains_plotly,
    customer_purchase_timeline,
    decile_lift_plotly,
    feature_importance_plotly,
    horizontal_ranking_chart,
    lorenz_concentration_chart,
    sales_timeseries_chart,
    segment_churn_distribution,
    segment_clv_distribution,
    segment_revenue_bar,
    segment_treemap,
    weekday_hour_heatmap_chart,
)
from app.components import (
    inject_theme_css,
    load_predictions,
    load_report_csv,
    load_transactions_with_segments,
    render_hero_header,
    render_segment_cards,
    render_sidebar_portfolio_filters,
    require_predictions,
    require_transactions,
    show_customer_table,
)
from app.insights import (
    CHURN_RISK_THRESHOLD,
    SEGMENT_ORDER,
    aggregate_sales_timeseries,
    compute_cumulative_gains_from_deciles,
    compute_lorenz_curve,
    compute_period_comparison,
    compute_portfolio_kpis,
    compute_segment_summary,
    compute_top_dimensions,
    compute_weekday_hour_heatmap,
    drilldown_period,
    filter_transactions,
    format_ordinal,
    generate_hero_headline,
    generate_portfolio_takeaways,
    get_customer_history,
    get_customer_profile,
    get_save_these_first,
    summarize_model_comparison,
)
from src.predict import predict_file

st.set_page_config(
    page_title="Customer Value | Retention Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme_css()

PAGES = [
    "Portfolio overview",
    "Customer segments",
    "Customer lookup",
    "Model performance",
    "Sales activity explorer",
    "Score a file",
    "How to read this",
]


def _delta_text(delta: float | None) -> str | None:
    if delta is None:
        return None
    return f"{delta:+.1%}"


def page_portfolio_overview(filtered_preds: pd.DataFrame, actions_df: pd.DataFrame) -> None:
    kpis = compute_portfolio_kpis(filtered_preds)
    render_hero_header(
        eyebrow="Six-month portfolio forecast",
        title="Customer value & retention portfolio",
        subtitle="Customer-level six-month revenue predictions paired with 90-day inactivity risk.",
        headline_insight=generate_hero_headline(kpis),
    )
    if not require_predictions(filtered_preds):
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Forecast revenue",
        f"£{kpis['total_clv']:,.0f}",
        help="Sum of predicted six-month transaction revenue across the filtered customers.",
    )
    c1.caption(f"Across {kpis['customer_count']:,} scored customers")

    c2.metric(
        "Average 6m CLV",
        f"£{kpis['avg_clv']:,.0f}",
        help="Mean predicted six-month revenue per customer.",
    )
    c2.caption(f"Median customer: £{kpis['median_clv']:,.0f}")

    c3.metric(
        "Top 20% share",
        f"{kpis['top_20_share']:.1%}",
        help="Share of forecast revenue concentrated in the highest-CLV 20% of customers.",
    )
    c3.caption(f"Top 10% drive {kpis['top_10_share']:.1%}")

    c4.metric(
        "At-risk (≥50%)",
        f"{kpis['high_risk_count']:,}",
        help="Customers with predicted 90-day inactivity probability >= 0.50.",
    )
    c4.caption(f"accounts · £{kpis['high_risk_revenue']:,.0f} ({kpis['high_risk_revenue_share']:.1%})")

    c5.metric(
        "Top segment",
        f"{kpis['top_segment_share']:.1%}",
        help="Share of forecast revenue held by the largest segment.",
    )
    c5.caption(f"{kpis['top_segment']} (£{kpis['top_segment_revenue']:,.0f})")

    st.markdown("#### Key takeaways")
    takeaways = generate_portfolio_takeaways(filtered_preds, kpis)
    with st.container(border=True):
        for item in takeaways:
            st.markdown(f"- {item}")

    seg_summary = compute_segment_summary(filtered_preds, actions_df=actions_df, sort_by="revenue")
    left, right = st.columns([1.25, 1.0])
    with left:
        st.subheader("Forecast value by customer group")
        st.plotly_chart(segment_revenue_bar(seg_summary), use_container_width=True)
        st.caption(
            "Sorted by six-month forecast revenue (y-axis starts at £0). "
            "Labels show total £ revenue and portfolio share per segment."
        )
    with right:
        st.subheader("Forecast revenue share by segment")
        st.plotly_chart(segment_treemap(seg_summary), use_container_width=True)
        st.caption("Tile area reflects six-month forecast revenue; hover for customer count and average risk.")

    lorenz_df = compute_lorenz_curve(filtered_preds)
    col_lorenz, col_scatter = st.columns([1.0, 1.25])
    with col_lorenz:
        st.subheader("Revenue concentration (Pareto / Lorenz)")
        st.plotly_chart(
            lorenz_concentration_chart(lorenz_df, kpis["top_10_share"], kpis["top_20_share"]),
            use_container_width=True,
        )
        st.caption(
            f"Top 10% of customers = {kpis['top_10_share']:.1%} of forecast revenue; "
            f"top 20% = {kpis['top_20_share']:.1%}."
        )

    with col_scatter:
        st.subheader("Predicted CLV vs. 90-day inactivity risk")
        st.plotly_chart(
            clv_churn_scatter(filtered_preds, kpis["clv_q80"], kpis["clv_q50"]),
            use_container_width=True,
        )
        st.caption(
            "Log-scale CLV plotted against 90-day inactivity probability, "
            "with dashed lines at the 50%/70% risk and 50th/80th CLV thresholds."
        )

    st.subheader("Save these first: high-value accounts at risk")
    st.caption(
        f"Customers with 90-day inactivity risk ≥ {CHURN_RISK_THRESHOLD:.0%}, "
        "ranked from highest six-month forecast value downward."
    )
    high_risk_all = get_save_these_first(filtered_preds, churn_threshold=CHURN_RISK_THRESHOLD, top_n=None)
    if high_risk_all.empty:
        st.info("No customers in the current filter selection have inactivity risk ≥ 50%.")
    else:
        show_customer_table(
            high_risk_all.head(50),
            download_label=f"Download all {len(high_risk_all):,} high-risk customers (CSV)",
            file_name="save_these_first_high_risk.csv",
        )


def page_customer_segments(filtered_preds: pd.DataFrame, actions_df: pd.DataFrame) -> None:
    render_hero_header(
        eyebrow="Retention strategy",
        title="Customer segments",
        subtitle="Profile cards, CLV/risk distributions, and recommended actions across the five priority groups.",
    )
    if not require_predictions(filtered_preds):
        return

    summary = compute_segment_summary(filtered_preds, actions_df=actions_df, sort_by="order")
    render_segment_cards(summary)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Forecast CLV distribution by segment")
        st.plotly_chart(segment_clv_distribution(filtered_preds), use_container_width=True)
        st.caption("Log-scale distribution of predicted six-month spend within each segment.")
    with right:
        st.subheader("90-day inactivity risk by segment")
        st.plotly_chart(segment_churn_distribution(filtered_preds), use_container_width=True)
        st.caption("Dashed thresholds mark the 50% At-Risk and 70% Hibernating cutoffs.")

    st.subheader("Inspect segment members")
    available = [s for s in SEGMENT_ORDER if s in filtered_preds["segment"].unique()]
    chosen = st.selectbox("Choose a segment", ["All filtered segments"] + available, key="seg_inspect")
    selected = (
        filtered_preds if chosen == "All filtered segments" else filtered_preds[filtered_preds["segment"] == chosen]
    ).sort_values("predicted_6m_clv", ascending=False)
    st.caption(f"{len(selected):,} customers · sorted by predicted 6-month CLV")
    show_customer_table(
        selected,
        download_label="Download segment table (CSV)",
        file_name=f"segment_{chosen.lower().replace(' ', '_')}.csv",
    )


def page_customer_lookup(all_preds: pd.DataFrame, tx_df: pd.DataFrame) -> None:
    render_hero_header(
        eyebrow="Account drill-down",
        title="Customer lookup",
        subtitle="Autocomplete by Customer ID to inspect forecast rank, risk, and historical purchase timeline.",
    )
    if not require_predictions(all_preds):
        return

    customer_ids = sorted(all_preds["Customer ID"].astype(str).unique().tolist())
    default_id = all_preds.sort_values("predicted_6m_clv", ascending=False)["Customer ID"].iloc[0]
    default_idx = customer_ids.index(default_id) if default_id in customer_ids else 0

    selected_id = st.selectbox(
        "Search or select a Customer ID",
        options=customer_ids,
        index=default_idx,
        help="Start typing any Customer ID to filter the list.",
    )
    profile = get_customer_profile(all_preds, selected_id)
    if not profile:
        st.warning("Customer ID not found in the scored portfolio.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted 6-month CLV", f"£{profile['predicted_6m_clv']:,.2f}")
    c1.caption(f"Rank #{profile['clv_rank']:,} of {profile['portfolio_size']:,}")

    c2.metric("90-day inactivity risk", f"{profile['churn_probability']:.1%}")
    c2.caption(f"{format_ordinal(profile['risk_percentile'])} risk percentile")

    c3.metric("Assigned segment", profile["segment"])
    c3.caption("Based on portfolio CLV & risk cutoffs")

    c4.metric("CLV percentile rank", f"{profile['clv_percentile']:.1f}%")
    c4.caption(f"Higher forecast than {profile['clv_percentile']:.1f}% of accounts")

    st.divider()
    st.subheader("Historical purchase activity (pre-2012 cleaned history)")
    if tx_df.empty:
        st.info(
            "Cleaned transaction history (`data/processed/transactions_clean.csv`) is not available, "
            "so historical orders and products cannot be shown."
        )
        return

    history = get_customer_history(tx_df, selected_id)
    if not history["has_history"]:
        st.info(f"No cleaned historical purchase rows were found for Customer {selected_id}.")
        return

    summary = history["summary"]
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Historical orders", f"{summary['total_orders']:,}")
    h1.caption(f"{summary['distinct_products']:,} distinct products")
    h2.metric("Historical spend", f"£{summary['total_spend']:,.2f}")
    h2.caption(f"Avg order value: £{summary['avg_order_value']:,.2f}")
    h3.metric("Country", summary["primary_country"])
    h3.caption(", ".join(summary["countries"][:3]))
    h4.metric("Active span", f"{summary['first_purchase'].strftime('%Y-%m-%d')}")
    h4.caption(f"Latest order: {summary['last_purchase'].strftime('%Y-%m-%d')}")

    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown("##### Purchase history timeline")
        st.plotly_chart(customer_purchase_timeline(history["timeline"]), use_container_width=True)
        st.caption("Each bar represents one historical invoice from transactions_clean.csv.")
    with right:
        st.markdown("##### Top purchased products")
        top_prod = history["top_products"].head(10)
        st.plotly_chart(
            horizontal_ranking_chart(
                top_prod,
                label_col="Description",
                value_col="Revenue",
                x_title="Historical spend (£)",
                color=PALETTE["secondary"],
                height=320,
            ),
            use_container_width=True,
        )
        st.caption("Top 10 products by cumulative historical revenue for this customer.")


def page_model_performance() -> None:
    render_hero_header(
        eyebrow="Time-aware validation",
        title="Model performance",
        subtitle="Chronological March–August 2011 holdout accuracy, decile lift, and feature importance.",
    )
    metrics_df = load_report_csv("model_comparison.csv")
    decile_df = load_report_csv("decile_lift.csv")
    importance_df = load_report_csv("feature_importance.csv")

    if metrics_df.empty:
        st.info("No `reports/model_comparison.csv` found. Run `python run_pipeline.py` to generate evaluation metrics.")
    else:
        annotated, verdict = summarize_model_comparison(metrics_df)
        st.success(verdict)

        display_df = annotated.copy()
        display_df["Status"] = display_df["is_best"].map(lambda b: "★ Best model" if b else "")
        display_df = display_df.drop(columns=["is_best"]).rename(
            columns={
                "model": "Model",
                "MAE": "Mean absolute error",
                "RMSE": "Root mean squared error",
                "R2": "R²",
                "Top 20% revenue captured": "Top 20% revenue captured",
            }
        )[["Status", "Model", "Mean absolute error", "Root mean squared error", "R²", "Top 20% revenue captured"]]

        def _highlight_best(row: pd.Series) -> list[str]:
            if row["Status"]:
                return ["background-color: rgba(37, 99, 235, 0.14); font-weight: 600;"] * len(row)
            return [""] * len(row)

        styled = display_df.style.apply(_highlight_best, axis=1)
        st.dataframe(
            styled,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Mean absolute error": st.column_config.NumberColumn("Mean absolute error", format="£%.2f"),
                "Root mean squared error": st.column_config.NumberColumn("Root mean squared error", format="£%.2f"),
                "R²": st.column_config.NumberColumn("R²", format="%.3f"),
                "Top 20% revenue captured": st.column_config.NumberColumn(
                    "Top 20% revenue captured", format="percent"
                ),
            },
        )
        st.download_button(
            "Download model comparison (CSV)",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cumulative gains (BG/NBD + Gamma-Gamma)")
        if not decile_df.empty:
            gains_df = compute_cumulative_gains_from_deciles(decile_df)
            st.plotly_chart(cumulative_gains_plotly(gains_df), use_container_width=True)
            st.caption("Share of actual holdout revenue captured as the targeted customer list expands by predicted CLV.")
        else:
            st.info("`reports/decile_lift.csv` not found.")

    with c2:
        st.subheader("Decile lift vs. portfolio average")
        if not decile_df.empty:
            st.plotly_chart(decile_lift_plotly(decile_df), use_container_width=True)
            st.caption("Decile 10 (top 10% by predicted CLV) delivers >6× the portfolio average spend.")
        else:
            st.info("`reports/decile_lift.csv` not found.")

    st.subheader("Feature importance (Gradient boosting permutation MAE)")
    if not importance_df.empty:
        st.plotly_chart(feature_importance_plotly(importance_df), use_container_width=True)
        st.caption(
            "Mean absolute error reduction when each feature is preserved vs. permuted on the holdout set."
        )
    else:
        st.info("`reports/feature_importance.csv` not found.")

    gains_png = ROOT / "reports" / "figures" / "cumulative_gains.png"
    if gains_png.exists():
        with st.expander("View static 3-model cumulative gains figure from pipeline"):
            st.image(str(gains_png), use_container_width=True)


def page_sales_explorer(tx_df: pd.DataFrame) -> None:
    render_hero_header(
        eyebrow="Historical transactions only",
        title="Sales activity explorer",
        subtitle="Explore cleaned historical revenue, orders, seasonality, and daily/weekly drill-downs.",
    )
    st.info(
        "**Historical activity note:** This page summarizes cleaned past transactions from "
        "`transactions_clean.csv` (**December 2009 – December 2011**). It is **not** a forecast — "
        "customer forecasts are six-month totals on the Portfolio overview page."
    )
    if not require_transactions(tx_df):
        return

    min_date = tx_df["InvoiceDate"].min().date()
    max_date = tx_df["InvoiceDate"].max().date()
    all_countries = sorted(tx_df["Country"].dropna().unique().tolist())
    all_segments = [s for s in SEGMENT_ORDER if s in tx_df["segment"].unique()] + (
        ["Post-cutoff / New"] if "Post-cutoff / New" in tx_df["segment"].unique() else []
    )

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.3, 0.9, 1.3, 1.3])
        default_start = max(min_date, pd.Timestamp("2010-12-01").date())
        date_range = f1.date_input(
            "Historical date range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range if not isinstance(date_range, (tuple, list)) else date_range[0]
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        granularity = f2.radio("Granularity", ["Day", "Week", "Month"], index=1, horizontal=True)
        selected_countries = f3.multiselect(
            "Country filter (empty = all)",
            options=all_countries,
            default=[],
        )
        selected_segments = f4.multiselect(
            "Customer segment filter",
            options=all_segments,
            default=all_segments,
        )

        o1, o2, o3 = st.columns([1.2, 1.0, 1.0])
        comp_choice = o1.radio(
            "Comparison window",
            ["Previous equal-length range", "Same range 1 year earlier"],
            horizontal=True,
        )
        show_anomalies = o2.checkbox("Highlight anomalies (spikes/dips)", value=True)
        highlight_q4 = o3.checkbox("Highlight Q4 / Christmas peak", value=True)

    comp_mode = "yoy" if "year" in comp_choice else "previous"
    filtered_tx = filter_transactions(
        tx_df,
        start_date=start_date,
        end_date=end_date,
        countries=selected_countries or None,
        segments=selected_segments,
    )

    if filtered_tx.empty:
        st.warning(
            "No historical transactions match the selected date range, countries, and segments. "
            "Try widening the date range or clearing filters."
        )
    else:
        comp = compute_period_comparison(
            tx_df,
            start_date=start_date,
            end_date=end_date,
            mode=comp_mode,
            countries=selected_countries or None,
            segments=selected_segments,
        )
        curr = comp["current"]
        deltas = comp["delta_pct"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Historical revenue", f"£{curr['revenue']:,.0f}", delta=_delta_text(deltas["revenue"]))
        k2.metric("Orders", f"{curr['orders']:,}", delta=_delta_text(deltas["orders"]))
        k3.metric("Active customers", f"{curr['active_customers']:,}", delta=_delta_text(deltas["active_customers"]))
        k4.metric("Average order value", f"£{curr['aov']:,.2f}", delta=_delta_text(deltas["aov"]))

        if comp["has_baseline"]:
            st.caption(
                f"Period comparison against {comp['label'].lower()}: "
                f"{comp['baseline_start']} to {comp['baseline_end']}."
            )
        else:
            st.caption(
                f"No transactions found in the comparison baseline "
                f"({comp['baseline_start']} to {comp['baseline_end']})."
            )

        ts_df = aggregate_sales_timeseries(filtered_tx, granularity=granularity)

        t1, t2 = st.columns(2)
        with t1:
            st.subheader(f"Historical revenue ({granularity.lower()})")
            st.plotly_chart(
                sales_timeseries_chart(
                    ts_df,
                    metric_col="revenue",
                    y_title="Revenue (£)",
                    is_currency=True,
                    show_anomalies=show_anomalies,
                    highlight_q4=highlight_q4,
                    color=PALETTE["primary"],
                ),
                use_container_width=True,
            )
            st.caption("Shaded Oct–Dec bands show Q4 holiday seasonality; markers flag statistical spikes and dips.")
        with t2:
            st.subheader(f"Historical orders ({granularity.lower()})")
            st.plotly_chart(
                sales_timeseries_chart(
                    ts_df,
                    metric_col="orders",
                    y_title="Distinct orders",
                    is_currency=False,
                    show_anomalies=show_anomalies,
                    highlight_q4=highlight_q4,
                    color=PALETTE["secondary"],
                ),
                use_container_width=True,
            )
            st.caption("Number of distinct non-cancelled invoices per period (y-axis starts at 0).")

        t3, t4 = st.columns(2)
        with t3:
            st.subheader(f"Active customers ({granularity.lower()})")
            st.plotly_chart(
                sales_timeseries_chart(
                    ts_df,
                    metric_col="active_customers",
                    y_title="Active customers",
                    is_currency=False,
                    show_anomalies=False,
                    highlight_q4=highlight_q4,
                    color="#6366F1",
                ),
                use_container_width=True,
            )
        with t4:
            st.subheader(f"Average order value ({granularity.lower()})")
            st.plotly_chart(
                sales_timeseries_chart(
                    ts_df,
                    metric_col="aov",
                    y_title="Average order value (£)",
                    is_currency=True,
                    show_anomalies=False,
                    highlight_q4=highlight_q4,
                    color=PALETTE["accent"],
                ),
                use_container_width=True,
            )

        st.subheader("Order volume by weekday and hour")
        heatmap_df = compute_weekday_hour_heatmap(filtered_tx)
        st.plotly_chart(weekday_hour_heatmap_chart(heatmap_df), use_container_width=True)
        st.caption("Weekday vs. hour-of-day heatmap of distinct invoices in the selected period.")

        top_prod, top_ctry = compute_top_dimensions(filtered_tx, top_n=10)
        d1, d2 = st.columns(2)
        with d1:
            st.subheader("Top 10 products (selected period)")
            st.plotly_chart(
                horizontal_ranking_chart(
                    top_prod,
                    label_col="Description",
                    value_col="revenue",
                    x_title="Historical revenue (£)",
                    color=PALETTE["primary"],
                    share_col="share",
                ),
                use_container_width=True,
            )
        with d2:
            st.subheader("Top 10 countries (selected period)")
            st.plotly_chart(
                horizontal_ranking_chart(
                    top_ctry,
                    label_col="Country",
                    value_col="revenue",
                    x_title="Historical revenue (£)",
                    color=PALETTE["secondary"],
                    share_col="share",
                ),
                use_container_width=True,
            )

        st.subheader("Download filtered historical data")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                f"Download {granularity.lower()} summary ({len(ts_df):,} rows)",
                data=ts_df.to_csv(index=False).encode("utf-8"),
                file_name=f"historical_sales_{granularity.lower()}_summary.csv",
                mime="text/csv",
            )
        with dl2:
            if len(filtered_tx) <= 100_000:
                st.download_button(
                    f"Download filtered transactions ({len(filtered_tx):,} rows)",
                    data=filtered_tx.to_csv(index=False).encode("utf-8"),
                    file_name="historical_transactions_filtered.csv",
                    mime="text/csv",
                )
            else:
                if st.button(f"Prepare full filtered CSV ({len(filtered_tx):,} rows)"):
                    st.download_button(
                        "Click to download prepared CSV",
                        data=filtered_tx.to_csv(index=False).encode("utf-8"),
                        file_name="historical_transactions_filtered.csv",
                        mime="text/csv",
                    )

    st.divider()
    st.subheader("Pick a single day or week (drill-down)")
    dd1, dd2 = st.columns([1, 1])
    drill_granularity = dd1.radio("Drill-down window", ["Day", "Week"], horizontal=True, key="drill_gran")
    drill_date = dd2.date_input(
        "Choose a date to inspect",
        value=min(end_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="drill_date",
    )
    filtered_for_drill = filter_transactions(
        tx_df,
        countries=selected_countries or None,
        segments=selected_segments,
    )
    drill = drilldown_period(
        full_transactions_df=tx_df,
        filtered_transactions_df=filtered_for_drill,
        target_date=drill_date,
        granularity=drill_granularity,
    )
    st.markdown(f"**Inspecting:** {drill['period_label']}")
    if not drill["has_data"]:
        st.info(drill["empty_reason"])
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Revenue", f"£{drill['revenue']:,.2f}")
        m2.metric("Orders", f"{drill['orders']:,}")
        m3.metric("Active customers", f"{drill['active_customers']:,}")
        m4.metric("New buyers", f"{drill['new_customers']:,}")
        m5.metric("Returning buyers", f"{drill['returning_customers']:,}")

        p_col, c_col = st.columns(2)
        with p_col:
            st.markdown("##### Top products in window")
            st.dataframe(
                drill["top_products"],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "revenue": st.column_config.NumberColumn("Revenue", format="£%.2f"),
                    "units": st.column_config.NumberColumn("Units", format="%d"),
                    "orders": st.column_config.NumberColumn("Orders", format="%d"),
                },
            )
        with c_col:
            st.markdown("##### Biggest customers in window")
            st.dataframe(
                drill["top_customers"],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Customer ID": st.column_config.TextColumn("Customer ID"),
                    "segment": st.column_config.TextColumn("Segment"),
                    "orders": st.column_config.NumberColumn("Orders", format="%d"),
                    "revenue": st.column_config.NumberColumn("Revenue", format="£%.2f"),
                },
            )


def page_score_file() -> None:
    render_hero_header(
        eyebrow="Batch inference",
        title="Score a transaction file",
        subtitle="Upload an Online Retail II-compatible CSV or Excel file to produce 6-month customer forecasts.",
    )
    st.info(
        "The file must include invoice, stock code, invoice date, quantity, price, and customer ID columns. "
        "Scoring trains the pipeline at the chosen cutoff using `src.predict.predict_file`."
    )
    uploaded = st.file_uploader("Transaction file", type=["csv", "xlsx"])
    cutoff = st.date_input(
        "Feature cutoff date",
        value=pd.Timestamp("2011-03-01").date(),
        help="Only transactions strictly before this date are used as customer features.",
    )
    if uploaded and st.button("Generate forecasts", type="primary"):
        data_dir = ROOT / "data" / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(uploaded.name).suffix.lower()
        temp_path = data_dir / f"upload_for_scoring{suffix}"
        try:
            temp_path.write_bytes(uploaded.getvalue())
            with st.spinner("Cleaning transactions and fitting scoring models…"):
                result = predict_file(temp_path, cutoff)
            st.success(f"Generated six-month forecasts for {len(result):,} customers.")
            show_customer_table(
                result,
                download_label="Download customer forecasts (CSV)",
                file_name="customer_predictions.csv",
            )
        except Exception as exc:
            st.error(
                f"Could not score `{uploaded.name}`. Verify the file columns and cutoff date ({exc})."
            )
        finally:
            temp_path.unlink(missing_ok=True)


def page_how_to_read() -> None:
    render_hero_header(
        eyebrow="Documentation & glossary",
        title="How to read this dashboard",
        subtitle="Pipeline workflow, segment rules, metric definitions, and important caveats.",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Pipeline workflow")
        st.markdown(
            "1. **Clean:** remove missing customer IDs, exact duplicates, `C`-prefixed cancellations, and non-positive prices/quantities; log every step in `cleaning_audit.csv`.\n"
            "2. **Engineer features:** compute recency, frequency, monetary spend, tenure, basket value, order intervals, distinct SKUs, return rate, and Q4 share strictly before the cutoff.\n"
            "3. **Forecast:** fit BG/NBD + Gamma-Gamma for six-month expected revenue and a balanced Random Forest for 90-day inactivity probability.\n"
            "4. **Segment:** assign customers into five priority groups for retention review."
        )
    with c2:
        st.subheader("Segment assignment rules")
        st.markdown(
            "- **Champions:** top 20% predicted CLV (≥80th percentile) and inactivity risk < 50%.\n"
            "- **Loyal:** above-median CLV (50th–80th percentile) and inactivity risk < 50%.\n"
            "- **At-Risk:** top 20% predicted CLV (≥80th percentile) and inactivity risk ≥ 50%.\n"
            "- **Hibernating:** inactivity risk ≥ 70% (and not in top 20% CLV).\n"
            "- **Potential:** remaining accounts."
        )

    st.subheader("Glossary")
    glossary = [
        ("Predicted 6-month CLV", "Model-estimated gross transaction revenue over the six months following the cutoff (not net profit)."),
        ("90-day inactivity risk", "Estimated probability that a customer places zero orders in the 90 days after the cutoff."),
        ("BG/NBD + Gamma-Gamma", "Probabilistic customer-base model pairing purchase-rate dynamics (BG/NBD) with expected order spend (Gamma-Gamma)."),
        ("Pareto / Lorenz curve", "Ranks customers from highest to lowest CLV to show what share of forecast revenue comes from the top X% of accounts."),
        ("Decile lift", "Average actual holdout revenue in a predicted-CLV decile divided by the overall portfolio average."),
        ("Average order value (AOV)", "Total historical transaction revenue divided by the number of distinct non-cancelled invoices."),
        ("Q4 seasonality", "October–December trading period, historically the peak revenue quarter in UCI Online Retail II."),
        ("Cutoff & holdout", "Features use transactions strictly before the cutoff date; holdout metrics test against actual revenue after the cutoff."),
    ]
    rows_md = "\n".join(f"| **{term}** | {definition} |" for term, definition in glossary)
    st.markdown(f"| Term | Definition |\n| --- | --- |\n{rows_md}")

    st.warning(
        "This dataset covers historical transactions from a single UK retailer ending in December 2011. "
        "Transaction revenue is not contribution margin. Treat forecasts as decision support and validate "
        "campaigns with current data and controlled holdout tests."
    )


def main() -> None:
    st.sidebar.markdown("### Customer Value Engine")
    st.sidebar.caption("6-month CLV forecasts · historical sales explorer")
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    st.sidebar.divider()

    predictions = load_predictions()
    actions_df = load_report_csv("segment_actions.csv")

    if page in ("Portfolio overview", "Customer segments"):
        filtered_preds = render_sidebar_portfolio_filters(predictions)
        st.sidebar.divider()
    else:
        filtered_preds = predictions

    st.sidebar.caption("Data source: UCI Online Retail II (historical through Dec 2011)")

    if page == "Portfolio overview":
        page_portfolio_overview(filtered_preds, actions_df)
    elif page == "Customer segments":
        page_customer_segments(filtered_preds, actions_df)
    elif page == "Customer lookup":
        tx_df = load_transactions_with_segments()
        page_customer_lookup(predictions, tx_df)
    elif page == "Model performance":
        page_model_performance()
    elif page == "Sales activity explorer":
        tx_df = load_transactions_with_segments()
        page_sales_explorer(tx_df)
    elif page == "Score a file":
        page_score_file()
    else:
        page_how_to_read()


if __name__ == "__main__":
    main()
