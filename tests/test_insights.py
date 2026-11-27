"""Unit tests for pure dashboard aggregation, insight, and drill-down helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from app.insights import (
    aggregate_sales_timeseries,
    attach_customer_segments,
    compute_cumulative_gains_from_deciles,
    compute_lorenz_curve,
    compute_period_comparison,
    compute_portfolio_kpis,
    compute_segment_summary,
    compute_top_dimensions,
    compute_weekday_hour_heatmap,
    drilldown_period,
    filter_predictions,
    filter_transactions,
    format_ordinal,
    generate_hero_headline,
    generate_portfolio_takeaways,
    get_customer_history,
    get_customer_profile,
    get_save_these_first,
    summarize_model_comparison,
)


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Customer ID": ["1001.0", "1002", "1003", "1004", "1005"],
            "predicted_6m_clv": [1000.0, 500.0, 200.0, 150.0, 50.0],
            "churn_probability": [0.10, 0.25, 0.60, 0.55, 0.85],
            "segment": ["Champions", "Loyal", "At-Risk", "Potential", "Hibernating"],
        }
    )


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": ["INV1", "INV2", "INV3", "INV4", "INV5", "INV6", "INV7", "INV8"],
            "StockCode": ["A1", "A2", "A1", "A3", "A2", "A1", "A4", "A4"],
            "Description": ["Mug", "Teapot", "Mug", "Candle", "Teapot", "Mug", "Tray", "Tray"],
            "Quantity": [2, 1, 5, 3, 2, 10, 1, 1],
            "InvoiceDate": pd.to_datetime(
                [
                    "2010-10-04 09:30:00",
                    "2010-10-11 11:00:00",
                    "2011-10-03 10:15:00",
                    "2011-10-03 14:00:00",
                    "2011-10-10 12:00:00",
                    "2011-10-17 15:00:00",
                    "2011-10-24 16:00:00",
                    "2011-10-31 16:30:00",
                ]
            ),
            "Price": [10.0, 50.0, 10.0, 20.0, 50.0, 100.0, 30.0, 35.0],
            "Customer ID": ["1001", "1002", "1001", "1003", "1002", "1001", "9999", "1004"],
            "Country": [
                "United Kingdom",
                "France",
                "United Kingdom",
                "United Kingdom",
                "France",
                "United Kingdom",
                "Germany",
                "United Kingdom",
            ],
            "Revenue": [20.0, 50.0, 50.0, 60.0, 100.0, 1000.0, 30.0, 35.0],
        }
    )


def test_filter_predictions_and_kpis():
    preds = _sample_predictions()
    kpis = compute_portfolio_kpis(preds)
    assert kpis["customer_count"] == 5
    assert kpis["total_clv"] == pytest.approx(1900.0)
    assert kpis["top_20_share"] == pytest.approx(1000.0 / 1900.0)
    assert kpis["high_risk_count"] == 3
    assert kpis["high_risk_revenue"] == pytest.approx(400.0)
    assert kpis["top_segment"] == "Champions"

    filtered = filter_predictions(preds, segments=["Champions", "At-Risk"], churn_range=(0.0, 0.7))
    assert list(filtered["Customer ID"]) == ["1001", "1003"]

    empty_kpis = compute_portfolio_kpis(preds.iloc[0:0])
    assert empty_kpis["customer_count"] == 0
    assert "No customers" in generate_hero_headline(empty_kpis)


def test_segment_summary_lorenz_and_save_first():
    preds = _sample_predictions()
    summary = compute_segment_summary(preds, sort_by="revenue")
    assert summary["revenue_share"].sum() == pytest.approx(1.0)
    assert summary.iloc[0]["segment"] == "Champions"

    lorenz = compute_lorenz_curve(preds)
    assert lorenz["customer_pct"].iloc[0] == 0.0
    assert lorenz["revenue_pct"].iloc[0] == 0.0
    assert lorenz["customer_pct"].iloc[-1] == pytest.approx(100.0)
    assert lorenz["revenue_pct"].iloc[-1] == pytest.approx(100.0)
    assert lorenz["revenue_pct"].is_monotonic_increasing

    save_first = get_save_these_first(preds, churn_threshold=0.5, top_n=10)
    assert list(save_first["Customer ID"]) == ["1003", "1004", "1005"]

    takeaways = generate_portfolio_takeaways(preds, compute_portfolio_kpis(preds))
    assert 3 <= len(takeaways) <= 5


def test_customer_profile_and_history():
    preds = _sample_predictions()
    tx = attach_customer_segments(_sample_transactions(), preds)
    profile = get_customer_profile(preds, "1001")
    assert profile is not None
    assert profile["clv_rank"] == 1
    assert profile["clv_percentile"] == pytest.approx(100.0)

    hist = get_customer_history(tx, "1001")
    assert hist["has_history"] is True
    assert hist["summary"]["total_orders"] == 3
    assert hist["summary"]["total_spend"] == pytest.approx(1070.0)
    assert hist["summary"]["primary_country"] == "United Kingdom"

    missing_hist = get_customer_history(tx, "0000")
    assert missing_hist["has_history"] is False


def test_model_comparison_and_decile_gains():
    metrics = pd.DataFrame(
        {
            "model": ["Historical average", "BG/NBD + Gamma-Gamma", "Gradient boosting"],
            "MAE": [573.3, 558.3, 1227.4],
            "RMSE": [2004.5, 1661.3, 2916.7],
            "R2": [0.68, 0.78, 0.33],
            "Top 20% revenue captured": [0.76, 0.78, 0.76],
        }
    )
    annotated, verdict = summarize_model_comparison(metrics)
    assert bool(annotated.loc[annotated["model"] == "BG/NBD + Gamma-Gamma", "is_best"].iloc[0])
    assert "BG/NBD + Gamma-Gamma" in verdict

    deciles = pd.DataFrame(
        {
            "decile": [10, 5, 1],
            "count": [10, 10, 10],
            "sum": [700.0, 200.0, 100.0],
            "mean": [70.0, 20.0, 10.0],
            "lift_vs_average": [2.1, 0.6, 0.3],
        }
    )
    gains = compute_cumulative_gains_from_deciles(deciles)
    assert gains["customer_pct"].iloc[0] == 0.0
    assert gains["revenue_pct"].iloc[-1] == pytest.approx(100.0)


def test_sales_explorer_aggregations_and_drilldown():
    preds = _sample_predictions()
    tx = attach_customer_segments(_sample_transactions(), preds)
    assert "Post-cutoff / New" in set(tx["segment"])

    oct_2011 = filter_transactions(tx, start_date="2011-10-01", end_date="2011-10-31")
    ts = aggregate_sales_timeseries(oct_2011, granularity="Week")
    assert not ts.empty
    assert ts["is_q4"].all()
    assert (ts["revenue"] >= 0).all()
    assert "High spike" in set(ts["anomaly"])

    comp_yoy = compute_period_comparison(tx, "2011-10-01", "2011-10-15", mode="yoy")
    assert comp_yoy["has_baseline"] is True
    assert comp_yoy["baseline"]["revenue"] == pytest.approx(70.0)

    heat = compute_weekday_hour_heatmap(oct_2011)
    assert "Monday" in heat.index
    assert int(heat.to_numpy().sum()) == 6

    top_prod, top_ctry = compute_top_dimensions(oct_2011, top_n=5)
    assert top_prod.iloc[0]["Description"] == "Mug"
    assert top_ctry.iloc[0]["Country"] == "United Kingdom"

    # Day with data: 2011-10-03 has customer 1001 (returning from 2010-10-04) and 1003 (new)
    day_hit = drilldown_period(tx, tx, "2011-10-03", granularity="Day")
    assert day_hit["has_data"] is True
    assert day_hit["orders"] == 2
    assert day_hit["new_customers"] == 1
    assert day_hit["returning_customers"] == 1

    # Day without data: Saturday 2011-10-08
    day_miss = drilldown_period(tx, tx, "2011-10-08", granularity="Day")
    assert day_miss["has_data"] is False
    assert "Saturday" in day_miss["empty_reason"]


def test_plotly_charts_zero_baseline_and_build():
    from app.charts import (
        clv_churn_scatter,
        lorenz_concentration_chart,
        sales_timeseries_chart,
        segment_revenue_bar,
    )

    preds = _sample_predictions()
    kpis = compute_portfolio_kpis(preds)
    seg_summary = compute_segment_summary(preds, sort_by="revenue")
    bar_fig = segment_revenue_bar(seg_summary)
    assert bar_fig.layout.yaxis.range[0] == 0

    lorenz_fig = lorenz_concentration_chart(
        compute_lorenz_curve(preds), kpis["top_10_share"], kpis["top_20_share"]
    )
    assert lorenz_fig.layout.yaxis.range[0] == 0

    scatter_fig = clv_churn_scatter(preds, kpis["clv_q80"], kpis["clv_q50"])
    assert len(scatter_fig.data) == 5

    tx = attach_customer_segments(_sample_transactions(), preds)
    ts = aggregate_sales_timeseries(tx, granularity="Week")
    ts_fig = sales_timeseries_chart(ts, "revenue", "Revenue (£)")
    assert ts_fig.layout.yaxis.range[0] == 0


def test_format_ordinal():
    assert format_ordinal(1) == "1st"
    assert format_ordinal(2) == "2nd"
    assert format_ordinal(3) == "3rd"
    assert format_ordinal(11) == "11th"
    assert format_ordinal(13) == "13th"
    assert format_ordinal(23) == "23rd"

