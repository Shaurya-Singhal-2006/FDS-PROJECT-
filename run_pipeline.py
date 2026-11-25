"""Run cleaning, analysis, time-aware model evaluation, and final scoring."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_prep import OUT, ROOT, clean_transactions, load_raw
from src.evaluate import decile_lift, gains_chart, permutation_importance_table
from src.features import FEATURES
from src.models import score_models
from src.predict import assign_segments, train_engine


def make_exploration_figures(transactions, figure_dir):
    """Create descriptive charts from cleaned, customer-linked purchases."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    monthly = transactions.set_index("InvoiceDate").resample("MS").Revenue.sum()
    monthly.plot(figsize=(8, 4), title="Monthly revenue")
    plt.ylabel("Revenue (£)")
    plt.tight_layout()
    plt.savefig(figure_dir / "monthly_revenue.png", dpi=150)
    plt.close()

    countries = transactions.groupby("Country").Revenue.sum().nlargest(10).sort_values()
    countries.plot.barh(figsize=(7, 5), title="Top countries by revenue")
    plt.tight_layout()
    plt.savefig(figure_dir / "top_countries.png", dpi=150)
    plt.close()

    products = transactions.groupby("Description").Revenue.sum().nlargest(10).sort_values()
    products.plot.barh(figsize=(8, 5), title="Top products by revenue")
    plt.tight_layout()
    plt.savefig(figure_dir / "top_products.png", dpi=150)
    plt.close()

    orders = transactions.groupby("Customer ID").Invoice.nunique()
    orders.value_counts().sort_index().head(25).plot.bar(
        figsize=(8, 4), title="Order frequency distribution"
    )
    plt.tight_layout()
    plt.savefig(figure_dir / "order_frequency.png", dpi=150)
    plt.close()

    customer_revenue = transactions.groupby("Customer ID").Revenue.sum().sort_values(
        ascending=False
    )
    share = customer_revenue.cumsum() / customer_revenue.sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(share) + 1) / len(share), share)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(
        title="Customer revenue concentration",
        xlabel="Share of customers (ranked)",
        ylabel="Cumulative revenue share",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "pareto_curve.png", dpi=150)
    plt.close(fig)


def write_segment_reports(customers, report_dir):
    """Save segment summaries, suggested actions, and an illustrative ROI case."""
    segments = customers.groupby("segment").agg(
        customers=("Customer ID", "count"),
        predicted_revenue=("predicted_6m_clv", "sum"),
        avg_churn=("churn_probability", "mean"),
    )
    segments["revenue_share"] = (
        segments.predicted_revenue / segments.predicted_revenue.sum()
    )
    segments.to_csv(report_dir / "segment_summary.csv")

    actions = {
        "Champions": "VIP service, early access and a replenishment reminder; avoid routine discounts.",
        "Loyal": "Personalized cross-sell and relevant replenishment messages.",
        "Potential": "Low-cost onboarding and second-purchase nudges.",
        "At-Risk": "Personalized win-back offer, prioritized by value and tested against a holdout.",
        "Hibernating": "Capped low-cost reactivation test; suppress non-responders.",
    }
    segments.reset_index().assign(
        recommended_action=lambda frame: frame.segment.map(actions)
    ).to_csv(report_dir / "segment_actions.csv", index=False)

    # This planning scenario uses assumptions, not measured campaign effects.
    selected = customers[customers.segment.isin(["Champions", "At-Risk"])]
    scenario = {
        "target_segments": "Champions + At-Risk",
        "customers": len(selected),
        "forecast_revenue": selected.predicted_6m_clv.sum(),
        "assumed_margin": 0.30,
        "assumed_incremental_retention": 0.10,
        "assumed_contact_cost_gbp": 1.0,
    }
    scenario["estimated_incremental_profit_gbp"] = (
        scenario["forecast_revenue"]
        * scenario["assumed_margin"]
        * scenario["assumed_incremental_retention"]
    )
    scenario["estimated_campaign_cost_gbp"] = (
        scenario["customers"] * scenario["assumed_contact_cost_gbp"]
    )
    scenario["estimated_net_benefit_gbp"] = (
        scenario["estimated_incremental_profit_gbp"]
        - scenario["estimated_campaign_cost_gbp"]
    )
    scenario["estimated_roi_percent"] = (
        100
        * scenario["estimated_net_benefit_gbp"]
        / max(scenario["estimated_campaign_cost_gbp"], 1)
    )
    pd.DataFrame([scenario]).to_csv(report_dir / "roi_scenario.csv", index=False)


def main():
    """Build every generated data, model, and report artifact."""
    np.random.seed(42)
    transactions, audit = clean_transactions(load_raw())
    returns_file = OUT / "transactions_returns.csv"
    returns = pd.read_csv(returns_file) if returns_file.exists() else None

    figure_dir = ROOT / "reports" / "figures"
    make_exploration_figures(transactions, figure_dir)

    # Tune on the earlier snapshot; evaluate at the March 2011 cutoff against
    # actual revenue from March through August 2011.
    cutoff = pd.Timestamp("2011-03-01")
    customers, bundle, predictions = train_engine(
        transactions, cutoff, return_evaluation=True
    )
    metrics = score_models(customers.future_revenue, predictions)
    metrics.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

    best_prediction = predictions["BG/NBD + Gamma-Gamma"]
    decile_lift(customers.future_revenue, best_prediction).to_csv(
        ROOT / "reports" / "decile_lift.csv", index=False
    )
    gains_chart(customers.future_revenue, predictions, figure_dir / "cumulative_gains.png")
    importance = permutation_importance_table(
        bundle["gradient_boosting"], customers[FEATURES], customers.future_revenue
    )
    importance.to_csv(ROOT / "reports" / "feature_importance.csv", index=False)

    features = customers[FEATURES]
    customers["predicted_6m_clv"] = np.maximum(
        bundle["regressor"].predict(customers), 0
    )
    classes = list(bundle["churn"][-1].classes_)
    customers["churn_probability"] = (
        bundle["churn"].predict_proba(features)[:, classes.index(1)]
        if 1 in classes
        else 0.0
    )
    customers["segment"] = assign_segments(
        customers.predicted_6m_clv, customers.churn_probability
    )
    prediction_columns = [
        "Customer ID",
        "predicted_6m_clv",
        "churn_probability",
        "segment",
    ]
    customers[prediction_columns].to_csv(OUT / "customer_predictions.csv", index=False)

    write_segment_reports(customers, ROOT / "reports")
    print("Cleaning audit:\n", audit.to_string(index=False))
    print("\nModel comparison:\n", metrics.to_string(index=False))
    print("\nSegments:\n", customers.groupby("segment").size().to_string())


if __name__ == "__main__":
    main()
