"""Pure analytics, aggregations, and narrative insights for the CLV dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_ORDER = ["Champions", "Loyal", "Potential", "At-Risk", "Hibernating"]

SEGMENT_COLORS = {
    "Champions": "#2563EB",
    "Loyal": "#0D9488",
    "Potential": "#6366F1",
    "At-Risk": "#D97706",
    "Hibernating": "#DC2626",
    "Post-cutoff / New": "#94A3B8",
}

DEFAULT_SEGMENT_ACTIONS = {
    "Champions": "VIP service, early access and a replenishment reminder; avoid routine discounts.",
    "Loyal": "Personalized cross-sell and relevant replenishment messages.",
    "Potential": "Low-cost onboarding and second-purchase nudges.",
    "At-Risk": "Personalized win-back offer, prioritized by value and tested against a holdout.",
    "Hibernating": "Capped low-cost reactivation test; suppress non-responders.",
}

CHURN_RISK_THRESHOLD = 0.50
HIBERNATING_RISK_THRESHOLD = 0.70

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def format_ordinal(n: int) -> str:
    """Format an integer as an English ordinal (e.g. 1st, 2nd, 3rd, 11th)."""
    n_int = int(round(n))
    if 11 <= (n_int % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n_int % 10, "th")
    return f"{n_int}{suffix}"


def normalize_customer_ids(series: pd.Series) -> pd.Series:
    """Convert Customer ID values to clean string identifiers without trailing .0."""
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def filter_predictions(
    df: pd.DataFrame,
    segments: list[str] | None = None,
    churn_range: tuple[float, float] | None = None,
    clv_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Apply consistent sidebar portfolio filters to customer predictions."""
    if df.empty:
        return df.copy()
    out = df.copy()
    out["Customer ID"] = normalize_customer_ids(out["Customer ID"])
    if segments is not None:
        out = out[out["segment"].isin(segments)]
    if churn_range is not None:
        low, high = churn_range
        out = out[(out["churn_probability"] >= low) & (out["churn_probability"] <= high)]
    if clv_range is not None:
        low, high = clv_range
        out = out[(out["predicted_6m_clv"] >= low) & (out["predicted_6m_clv"] <= high)]
    return out.reset_index(drop=True)


def compute_portfolio_kpis(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute portfolio-level revenue, concentration, risk, and segment KPIs."""
    if df.empty:
        return {
            "customer_count": 0,
            "total_clv": 0.0,
            "avg_clv": 0.0,
            "median_clv": 0.0,
            "top_10_share": 0.0,
            "top_20_share": 0.0,
            "high_risk_count": 0,
            "high_risk_share": 0.0,
            "high_risk_revenue": 0.0,
            "high_risk_revenue_share": 0.0,
            "top_segment": "N/A",
            "top_segment_revenue": 0.0,
            "top_segment_share": 0.0,
            "clv_q80": 0.0,
            "clv_q50": 0.0,
        }

    n = len(df)
    sorted_clv = df["predicted_6m_clv"].sort_values(ascending=False).to_numpy(dtype=float)
    total_clv = float(sorted_clv.sum())
    k10 = max(1, int(np.ceil(n * 0.10)))
    k20 = max(1, int(np.ceil(n * 0.20)))
    top_10_share = float(sorted_clv[:k10].sum() / total_clv) if total_clv > 0 else 0.0
    top_20_share = float(sorted_clv[:k20].sum() / total_clv) if total_clv > 0 else 0.0

    high_risk_mask = df["churn_probability"] >= CHURN_RISK_THRESHOLD
    high_risk_count = int(high_risk_mask.sum())
    high_risk_revenue = float(df.loc[high_risk_mask, "predicted_6m_clv"].sum())

    seg_rev = df.groupby("segment")["predicted_6m_clv"].sum()
    top_segment = str(seg_rev.idxmax()) if not seg_rev.empty else "N/A"
    top_segment_revenue = float(seg_rev.max()) if not seg_rev.empty else 0.0

    return {
        "customer_count": n,
        "total_clv": total_clv,
        "avg_clv": float(df["predicted_6m_clv"].mean()),
        "median_clv": float(df["predicted_6m_clv"].median()),
        "top_10_share": top_10_share,
        "top_20_share": top_20_share,
        "high_risk_count": high_risk_count,
        "high_risk_share": float(high_risk_count / n) if n else 0.0,
        "high_risk_revenue": high_risk_revenue,
        "high_risk_revenue_share": float(high_risk_revenue / total_clv) if total_clv > 0 else 0.0,
        "top_segment": top_segment,
        "top_segment_revenue": top_segment_revenue,
        "top_segment_share": float(top_segment_revenue / total_clv) if total_clv > 0 else 0.0,
        "clv_q80": float(df["predicted_6m_clv"].quantile(0.80)),
        "clv_q50": float(df["predicted_6m_clv"].quantile(0.50)),
    }


def compute_segment_summary(
    df: pd.DataFrame,
    actions_df: pd.DataFrame | None = None,
    sort_by: str = "order",
) -> pd.DataFrame:
    """Aggregate customer counts, revenue shares, averages, and actions per segment."""
    columns = [
        "segment",
        "customers",
        "customer_share",
        "predicted_revenue",
        "revenue_share",
        "avg_clv",
        "avg_churn",
        "recommended_action",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    action_map = dict(DEFAULT_SEGMENT_ACTIONS)
    if actions_df is not None and not actions_df.empty and {"segment", "recommended_action"}.issubset(actions_df.columns):
        for _, row in actions_df.iterrows():
            action_map[str(row["segment"])] = str(row["recommended_action"])

    grouped = (
        df.groupby("segment", observed=True)
        .agg(
            customers=("Customer ID", "count"),
            predicted_revenue=("predicted_6m_clv", "sum"),
            avg_clv=("predicted_6m_clv", "mean"),
            avg_churn=("churn_probability", "mean"),
        )
        .reset_index()
    )
    total_customers = float(grouped["customers"].sum())
    total_revenue = float(grouped["predicted_revenue"].sum())
    grouped["customer_share"] = grouped["customers"] / total_customers if total_customers > 0 else 0.0
    grouped["revenue_share"] = grouped["predicted_revenue"] / total_revenue if total_revenue > 0 else 0.0
    grouped["recommended_action"] = grouped["segment"].map(lambda s: action_map.get(s, ""))

    if sort_by == "revenue":
        grouped = grouped.sort_values("predicted_revenue", ascending=False)
    else:
        order_index = {seg: idx for idx, seg in enumerate(SEGMENT_ORDER)}
        grouped["_ord"] = grouped["segment"].map(lambda s: order_index.get(s, 999))
        grouped = grouped.sort_values("_ord").drop(columns=["_ord"])

    return grouped[columns].reset_index(drop=True)


def compute_lorenz_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Build Pareto/Lorenz cumulative customer vs cumulative forecast revenue curve."""
    cols = ["customer_pct", "revenue_pct", "cumulative_customers", "cumulative_revenue"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    values = np.sort(df["predicted_6m_clv"].to_numpy(dtype=float))[::-1]
    n = len(values)
    total = float(values.sum())
    cum_rev = np.cumsum(values)
    cust_pct = np.concatenate(([0.0], np.arange(1, n + 1) / n * 100.0))
    rev_pct = np.concatenate(([0.0], (cum_rev / total * 100.0) if total > 0 else np.zeros(n)))
    cum_cust = np.concatenate(([0], np.arange(1, n + 1)))
    cum_rev_full = np.concatenate(([0.0], cum_rev))
    return pd.DataFrame(
        {
            "customer_pct": cust_pct,
            "revenue_pct": rev_pct,
            "cumulative_customers": cum_cust,
            "cumulative_revenue": cum_rev_full,
        }
    )


def get_save_these_first(
    df: pd.DataFrame,
    churn_threshold: float = CHURN_RISK_THRESHOLD,
    top_n: int | None = 50,
) -> pd.DataFrame:
    """Return high-risk customers ranked from highest predicted 6-month CLV down."""
    if df.empty:
        return pd.DataFrame(
            columns=["Customer ID", "predicted_6m_clv", "churn_probability", "segment"]
        )
    filtered = df[df["churn_probability"] >= churn_threshold].sort_values(
        ["predicted_6m_clv", "churn_probability"], ascending=[False, False]
    )
    if top_n is not None:
        filtered = filtered.head(top_n)
    return filtered[["Customer ID", "predicted_6m_clv", "churn_probability", "segment"]].reset_index(drop=True)


def generate_hero_headline(kpis: dict[str, float | int | str]) -> str:
    """Generate a single-line executive summary from current portfolio KPIs."""
    if not kpis["customer_count"]:
        return "No customers match the active filters — widen your sidebar filters to inspect the portfolio."
    return (
        f"Top 20% of customers account for {kpis['top_20_share']:.1%} of the "
        f"£{kpis['total_clv']:,.0f} six-month forecast, while {kpis['high_risk_count']:,} accounts "
        f"(£{kpis['high_risk_revenue']:,.0f}) carry ≥50% inactivity risk."
    )


def generate_portfolio_takeaways(
    df: pd.DataFrame,
    kpis: dict[str, float | int | str],
) -> list[str]:
    """Produce 4-5 data-driven plain-English portfolio insights."""
    if df.empty:
        return ["No customers match the current sidebar filters."]

    takeaways: list[str] = []
    takeaways.append(
        f"**High revenue concentration:** The top 10% of scored customers represent "
        f"**{kpis['top_10_share']:.1%}** of forecast 6-month revenue (top 20% = **{kpis['top_20_share']:.1%}**)."
    )

    seg_summary = compute_segment_summary(df, sort_by="revenue")
    if not seg_summary.empty:
        top_row = seg_summary.iloc[0]
        takeaways.append(
            f"**{top_row['segment']} lead portfolio value:** {int(top_row['customers']):,} customers "
            f"({top_row['customer_share']:.1%} of accounts) generate **£{top_row['predicted_revenue']:,.0f}** "
            f"(**{top_row['revenue_share']:.1%}** of forecast revenue), averaging **£{top_row['avg_clv']:,.0f}** per customer."
        )

    takeaways.append(
        f"**Inactivity exposure (≥50% risk):** **{kpis['high_risk_count']:,}** customers "
        f"({kpis['high_risk_share']:.1%} of the selection) represent **£{kpis['high_risk_revenue']:,.0f}** "
        f"({kpis['high_risk_revenue_share']:.1%} of forecast revenue) at elevated risk of 90-day inactivity."
    )

    at_risk = df[df["segment"] == "At-Risk"]
    if not at_risk.empty:
        takeaways.append(
            f"**Priority win-back target (`At-Risk`):** {len(at_risk):,} high-value accounts carry an average "
            f"inactivity risk of **{at_risk['churn_probability'].mean():.1%}** and average forecast value of "
            f"**£{at_risk['predicted_6m_clv'].mean():,.0f}** — ideal candidates for targeted retention outreach."
        )

    if kpis["median_clv"] > 0:
        ratio = kpis["avg_clv"] / kpis["median_clv"]
        takeaways.append(
            f"**Right-skewed spend distribution:** Mean forecast CLV (**£{kpis['avg_clv']:,.0f}**) is "
            f"**{ratio:.1f}×** the median (**£{kpis['median_clv']:,.0f}**), underscoring why tiered retention beats one-size-fits-all campaigns."
        )

    return takeaways[:5]


def get_customer_profile(predictions_df: pd.DataFrame, customer_id: str) -> dict | None:
    """Return forecast, risk, segment, and portfolio rank/percentiles for one customer."""
    if predictions_df.empty or not customer_id:
        return None
    df = predictions_df.copy()
    df["Customer ID"] = normalize_customer_ids(df["Customer ID"])
    cid = str(customer_id).strip()
    match = df[df["Customer ID"] == cid]
    if match.empty:
        return None

    row = match.iloc[0]
    n = len(df)
    clv = float(row["predicted_6m_clv"])
    churn = float(row["churn_probability"])
    clv_rank = int((df["predicted_6m_clv"] > clv).sum()) + 1
    clv_percentile = float((df["predicted_6m_clv"] <= clv).mean() * 100.0)
    risk_percentile = float((df["churn_probability"] <= churn).mean() * 100.0)
    return {
        "customer_id": cid,
        "predicted_6m_clv": clv,
        "churn_probability": churn,
        "segment": str(row["segment"]),
        "clv_rank": clv_rank,
        "portfolio_size": n,
        "clv_percentile": clv_percentile,
        "risk_percentile": risk_percentile,
    }


def get_customer_history(transactions_df: pd.DataFrame, customer_id: str) -> dict:
    """Extract historical purchase timeline, top products, and country for a customer."""
    empty_timeline = pd.DataFrame(columns=["Invoice", "InvoiceDate", "Revenue", "Quantity", "Items"])
    empty_products = pd.DataFrame(columns=["Description", "StockCode", "Revenue", "Quantity", "Orders"])
    if transactions_df.empty or not customer_id:
        return {
            "has_history": False,
            "summary": {},
            "timeline": empty_timeline,
            "top_products": empty_products,
        }

    cid = str(customer_id).strip()
    cust_tx = transactions_df[transactions_df["Customer ID"] == cid].copy()
    if cust_tx.empty:
        return {
            "has_history": False,
            "summary": {},
            "timeline": empty_timeline,
            "top_products": empty_products,
        }

    timeline = (
        cust_tx.groupby("Invoice", as_index=False)
        .agg(
            InvoiceDate=("InvoiceDate", "min"),
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            Items=("StockCode", "nunique"),
        )
        .sort_values("InvoiceDate")
        .reset_index(drop=True)
    )

    top_products = (
        cust_tx.assign(Description=cust_tx["Description"].fillna("Unknown product").astype(str))
        .groupby(["Description", "StockCode"], as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("Invoice", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )

    countries = cust_tx["Country"].dropna().astype(str).value_counts().index.tolist()
    total_spend = float(cust_tx["Revenue"].sum())
    total_orders = int(timeline["Invoice"].nunique())

    return {
        "has_history": True,
        "summary": {
            "total_orders": total_orders,
            "total_spend": total_spend,
            "total_units": int(cust_tx["Quantity"].sum()),
            "distinct_products": int(cust_tx["StockCode"].nunique()),
            "avg_order_value": total_spend / total_orders if total_orders > 0 else 0.0,
            "first_purchase": timeline["InvoiceDate"].min(),
            "last_purchase": timeline["InvoiceDate"].max(),
            "primary_country": countries[0] if countries else "Unknown",
            "countries": countries,
        },
        "timeline": timeline,
        "top_products": top_products,
    }


def summarize_model_comparison(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Identify the best model from holdout metrics and write a plain-English verdict."""
    if metrics_df.empty:
        return metrics_df.copy(), "Model comparison metrics are not available yet."

    df = metrics_df.copy()
    best_idx = df["MAE"].idxmin()
    best_row = df.loc[best_idx]
    df["is_best"] = df.index == best_idx

    others = df.drop(index=best_idx)
    if not others.empty:
        runner_up = others.loc[others["MAE"].idxmin()]
        mae_diff = float(runner_up["MAE"] - best_row["MAE"])
        verdict = (
            f"**{best_row['model']} is the best overall model** on the chronological March–August 2011 holdout: "
            f"it achieves the lowest forecast error (**£{best_row['MAE']:,.2f}** MAE, **£{best_row['RMSE']:,.2f}** RMSE — "
            f"£{mae_diff:,.2f} lower MAE than {runner_up['model']}), explains the most variance "
            f"(**{best_row['R2']:.3f} R²**), and captures **{best_row['Top 20% revenue captured']:.1%}** "
            f"of future revenue in the top-ranked 20% of customers."
        )
    else:
        verdict = (
            f"**{best_row['model']}** achieved **£{best_row['MAE']:,.2f}** MAE, "
            f"**{best_row['R2']:.3f} R²**, and **{best_row['Top 20% revenue captured']:.1%}** top-20% capture."
        )
    return df, verdict


def compute_cumulative_gains_from_deciles(decile_df: pd.DataFrame) -> pd.DataFrame:
    """Convert decile lift rows (decile 10 down to 1) into a cumulative gains curve."""
    cols = ["customer_pct", "revenue_pct", "random_pct", "decile", "lift_vs_average"]
    if decile_df.empty:
        return pd.DataFrame(columns=cols)

    ordered = decile_df.sort_values("decile", ascending=False).reset_index(drop=True)
    total_cust = float(ordered["count"].sum())
    total_rev = float(ordered["sum"].sum())
    cum_cust_pct = (ordered["count"].cumsum() / total_cust * 100.0) if total_cust > 0 else pd.Series(0.0, index=ordered.index)
    cum_rev_pct = (ordered["sum"].cumsum() / total_rev * 100.0) if total_rev > 0 else pd.Series(0.0, index=ordered.index)

    return pd.DataFrame(
        {
            "customer_pct": [0.0, *cum_cust_pct.tolist()],
            "revenue_pct": [0.0, *cum_rev_pct.tolist()],
            "random_pct": [0.0, *cum_cust_pct.tolist()],
            "decile": [np.nan, *ordered["decile"].tolist()],
            "lift_vs_average": [np.nan, *ordered["lift_vs_average"].tolist()],
        }
    )


def attach_customer_segments(
    transactions_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach scored segments to historical transactions; label unscored customers explicitly."""
    if transactions_df.empty:
        out = transactions_df.copy()
        out["segment"] = pd.Series(dtype=str)
        return out

    out = transactions_df.copy()
    out["Customer ID"] = normalize_customer_ids(out["Customer ID"])
    if predictions_df is None or predictions_df.empty:
        out["segment"] = "Post-cutoff / New"
        return out

    seg_lookup = (
        predictions_df[["Customer ID", "segment"]]
        .assign(**{"Customer ID": normalize_customer_ids(predictions_df["Customer ID"])})
        .drop_duplicates("Customer ID")
    )
    out = out.merge(seg_lookup, on="Customer ID", how="left")
    out["segment"] = out["segment"].fillna("Post-cutoff / New")
    return out


def filter_transactions(
    df: pd.DataFrame,
    start_date=None,
    end_date=None,
    countries: list[str] | None = None,
    segments: list[str] | None = None,
) -> pd.DataFrame:
    """Filter historical transactions by inclusive date range, countries, and segments."""
    if df.empty:
        return df.copy()
    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        start_ts = pd.Timestamp(start_date).normalize()
        mask &= df["InvoiceDate"] >= start_ts
    if end_date is not None:
        end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
        mask &= df["InvoiceDate"] < end_ts
    if countries:
        mask &= df["Country"].isin(countries)
    if segments and "segment" in df.columns:
        mask &= df["segment"].isin(segments)
    return df.loc[mask].copy()


def aggregate_sales_timeseries(df: pd.DataFrame, granularity: str = "Week") -> pd.DataFrame:
    """Aggregate historical sales by Day, Week, or Month and flag Q4 & statistical anomalies."""
    cols = ["period", "revenue", "orders", "active_customers", "aov", "is_q4", "anomaly"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    freq_map = {"Day": "D", "Week": "W-MON", "Month": "MS"}
    freq = freq_map.get(granularity, "W-MON")

    working = df.copy()
    if granularity == "Week":
        working["period"] = working["InvoiceDate"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    elif granularity == "Month":
        working["period"] = working["InvoiceDate"].dt.to_period("M").apply(lambda p: p.start_time)
    else:
        working["period"] = working["InvoiceDate"].dt.normalize()

    agg = (
        working.groupby("period", as_index=False)
        .agg(
            revenue=("Revenue", "sum"),
            orders=("Invoice", "nunique"),
            active_customers=("Customer ID", "nunique"),
        )
        .sort_values("period")
        .reset_index(drop=True)
    )
    agg["aov"] = np.where(agg["orders"] > 0, agg["revenue"] / agg["orders"], 0.0)
    agg["is_q4"] = agg["period"].dt.month.isin([10, 11, 12])

    agg["anomaly"] = "Normal"
    if len(agg) >= 5:
        median = float(agg["revenue"].median())
        mad = float(np.median(np.abs(agg["revenue"] - median)))
        scale = 1.4826 * mad
        if scale <= 0:
            scale = float(agg["revenue"].std(ddof=0))
        if scale > 0:
            z = (agg["revenue"] - median) / scale
            agg.loc[z >= 2.5, "anomaly"] = "High spike"
            agg.loc[z <= -2.0, "anomaly"] = "Low dip"

    return agg[cols]


def _period_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {"revenue": 0.0, "orders": 0, "active_customers": 0, "aov": 0.0}
    rev = float(df["Revenue"].sum())
    orders = int(df["Invoice"].nunique())
    custs = int(df["Customer ID"].nunique())
    return {
        "revenue": rev,
        "orders": orders,
        "active_customers": custs,
        "aov": rev / orders if orders > 0 else 0.0,
    }


def compute_period_comparison(
    transactions_df: pd.DataFrame,
    start_date,
    end_date,
    mode: str = "previous",
    countries: list[str] | None = None,
    segments: list[str] | None = None,
) -> dict:
    """Compare selected date window against the previous equal-length range or prior year."""
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    span_days = max(1, (end_ts - start_ts).days + 1)

    if mode == "yoy":
        base_start = start_ts - pd.DateOffset(years=1)
        base_end = end_ts - pd.DateOffset(years=1)
        label = "Same period prior year"
    else:
        base_end = start_ts - pd.Timedelta(days=1)
        base_start = base_end - pd.Timedelta(days=span_days - 1)
        label = f"Previous {span_days} day(s)"

    curr_df = filter_transactions(transactions_df, start_ts, end_ts, countries=countries, segments=segments)
    base_df = filter_transactions(transactions_df, base_start, base_end, countries=countries, segments=segments)

    curr = _period_metrics(curr_df)
    base = _period_metrics(base_df)
    has_baseline = not base_df.empty

    deltas = {}
    for key in ("revenue", "orders", "active_customers", "aov"):
        if has_baseline and base[key] > 0:
            deltas[key] = (curr[key] - base[key]) / base[key]
        else:
            deltas[key] = None

    return {
        "label": label,
        "current_start": start_ts.date(),
        "current_end": end_ts.date(),
        "baseline_start": base_start.date(),
        "baseline_end": base_end.date(),
        "has_baseline": has_baseline,
        "current": curr,
        "baseline": base,
        "delta_pct": deltas,
    }


def compute_weekday_hour_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Return a weekday x hour matrix of distinct order counts."""
    if df.empty:
        return pd.DataFrame(index=WEEKDAY_ORDER)

    working = df.copy()
    working["weekday"] = working["InvoiceDate"].dt.day_name()
    working["hour"] = working["InvoiceDate"].dt.hour
    grouped = (
        working.groupby(["weekday", "hour"], as_index=False)["Invoice"]
        .nunique()
        .pivot(index="weekday", columns="hour", values="Invoice")
        .fillna(0)
        .astype(int)
    )
    grouped = grouped.reindex(WEEKDAY_ORDER).fillna(0).astype(int)
    grouped = grouped.reindex(sorted(grouped.columns), axis=1)
    return grouped


def compute_top_dimensions(df: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute top products and top countries by historical revenue."""
    empty_prod = pd.DataFrame(columns=["Description", "revenue", "orders", "customers", "share"])
    empty_ctry = pd.DataFrame(columns=["Country", "revenue", "orders", "customers", "share"])
    if df.empty:
        return empty_prod, empty_ctry

    total_rev = float(df["Revenue"].sum())
    products = (
        df.assign(Description=df["Description"].fillna("Unknown product").astype(str))
        .groupby("Description", as_index=False)
        .agg(
            revenue=("Revenue", "sum"),
            orders=("Invoice", "nunique"),
            customers=("Customer ID", "nunique"),
        )
        .sort_values("revenue", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    products["share"] = products["revenue"] / total_rev if total_rev > 0 else 0.0

    countries = (
        df.assign(Country=df["Country"].fillna("Unknown").astype(str))
        .groupby("Country", as_index=False)
        .agg(
            revenue=("Revenue", "sum"),
            orders=("Invoice", "nunique"),
            customers=("Customer ID", "nunique"),
        )
        .sort_values("revenue", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    countries["share"] = countries["revenue"] / total_rev if total_rev > 0 else 0.0

    return products, countries


def drilldown_period(
    full_transactions_df: pd.DataFrame,
    filtered_transactions_df: pd.DataFrame,
    target_date,
    granularity: str = "Day",
) -> dict:
    """Inspect a single day or week in detail, including new vs returning customer split."""
    target_ts = pd.Timestamp(target_date).normalize()
    if granularity == "Week":
        win_start = target_ts - pd.Timedelta(days=target_ts.weekday())
        win_end = win_start + pd.Timedelta(days=6)
        label = f"Week of {win_start.strftime('%Y-%m-%d')} to {win_end.strftime('%Y-%m-%d')}"
    else:
        win_start = target_ts
        win_end = target_ts
        label = target_ts.strftime("%A, %Y-%m-%d")

    slice_df = filter_transactions(filtered_transactions_df, win_start, win_end)
    if slice_df.empty:
        reason = _explain_empty_period(full_transactions_df, win_start, win_end)
        return {
            "has_data": False,
            "period_label": label,
            "window_start": win_start.date(),
            "window_end": win_end.date(),
            "empty_reason": reason,
            "revenue": 0.0,
            "orders": 0,
            "active_customers": 0,
            "new_customers": 0,
            "returning_customers": 0,
            "aov": 0.0,
            "top_products": pd.DataFrame(columns=["Description", "revenue", "units", "orders"]),
            "top_customers": pd.DataFrame(columns=["Customer ID", "segment", "orders", "revenue"]),
        }

    # Determine first-ever purchase date from full clean history so new vs returning is accurate.
    first_seen = (
        full_transactions_df.groupby("Customer ID")["InvoiceDate"]
        .min()
        .dt.normalize()
    )
    active_ids = slice_df["Customer ID"].unique()
    first_dates = first_seen.reindex(active_ids)
    new_count = int(((first_dates >= win_start) & (first_dates <= win_end)).sum())
    returning_count = int(len(active_ids) - new_count)

    revenue = float(slice_df["Revenue"].sum())
    orders = int(slice_df["Invoice"].nunique())

    top_products = (
        slice_df.assign(Description=slice_df["Description"].fillna("Unknown product").astype(str))
        .groupby("Description", as_index=False)
        .agg(
            revenue=("Revenue", "sum"),
            units=("Quantity", "sum"),
            orders=("Invoice", "nunique"),
        )
        .sort_values("revenue", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    if "segment" in slice_df.columns:
        top_customers = (
            slice_df.groupby(["Customer ID", "segment"], as_index=False)
            .agg(orders=("Invoice", "nunique"), revenue=("Revenue", "sum"))
            .sort_values("revenue", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
    else:
        top_customers = (
            slice_df.groupby("Customer ID", as_index=False)
            .agg(orders=("Invoice", "nunique"), revenue=("Revenue", "sum"))
            .assign(segment="Unknown")
            .sort_values("revenue", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )[["Customer ID", "segment", "orders", "revenue"]]

    return {
        "has_data": True,
        "period_label": label,
        "window_start": win_start.date(),
        "window_end": win_end.date(),
        "empty_reason": "",
        "revenue": revenue,
        "orders": orders,
        "active_customers": int(len(active_ids)),
        "new_customers": new_count,
        "returning_customers": returning_count,
        "aov": revenue / orders if orders > 0 else 0.0,
        "top_products": top_products,
        "top_customers": top_customers,
    }


def _explain_empty_period(
    full_transactions_df: pd.DataFrame,
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
) -> str:
    """Give a clear reason why a drilled-down day or week has no matching orders."""
    if full_transactions_df.empty:
        return "No historical transaction records are loaded."

    min_date = full_transactions_df["InvoiceDate"].min().normalize()
    max_date = full_transactions_df["InvoiceDate"].max().normalize()
    if win_end < min_date or win_start > max_date:
        return (
            f"This selection falls outside the historical dataset range "
            f"({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})."
        )

    unfiltered = filter_transactions(full_transactions_df, win_start, win_end)
    if unfiltered.empty:
        if win_start == win_end and win_start.weekday() == 5:
            return (
                f"No customer orders were recorded on {win_start.strftime('%A, %Y-%m-%d')}. "
                "This UK wholesaler did not dispatch online orders on Saturdays."
            )
        return (
            "No valid customer-linked orders occurred in this window "
            "(for example, due to a weekend/holiday closure or non-trading day)."
        )

    return (
        "Transactions exist in the dataset for this date window, "
        "but none match your active country or segment filters."
    )
