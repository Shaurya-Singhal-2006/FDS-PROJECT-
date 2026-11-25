"""Run cleaning, analysis, time-aware model evaluation and final scoring."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_prep import load_raw, clean_transactions, OUT, ROOT
from src.features import FEATURES
from src.models import score_models
from src.evaluate import decile_lift, gains_chart, permutation_importance_table
from src.predict import train_engine, assign_segments

np.random.seed(42)
tx,audit=clean_transactions(load_raw())
returns=pd.read_csv(OUT/"transactions_returns.csv") if (OUT/"transactions_returns.csv").exists() else None
figdir=ROOT/"reports"/"figures"; figdir.mkdir(parents=True,exist_ok=True)
# EDA plots and business summaries on cleaned, identified positive purchases.
monthly=tx.set_index("InvoiceDate").resample("MS").Revenue.sum()
monthly.plot(figsize=(8,4),title="Monthly revenue"); plt.ylabel("Revenue (£)"); plt.tight_layout(); plt.savefig(figdir/"monthly_revenue.png",dpi=150); plt.close()
tx.groupby("Country").Revenue.sum().nlargest(10).sort_values().plot.barh(figsize=(7,5),title="Top countries by revenue"); plt.tight_layout(); plt.savefig(figdir/"top_countries.png",dpi=150); plt.close()
tx.groupby("Description").Revenue.sum().nlargest(10).sort_values().plot.barh(figsize=(8,5),title="Top products by revenue"); plt.tight_layout(); plt.savefig(figdir/"top_products.png",dpi=150); plt.close()
orders=tx.groupby("Customer ID").Invoice.nunique(); orders.value_counts().sort_index().head(25).plot.bar(figsize=(8,4),title="Order frequency distribution"); plt.tight_layout(); plt.savefig(figdir/"order_frequency.png",dpi=150); plt.close()
custrev=tx.groupby("Customer ID").Revenue.sum().sort_values(ascending=False); share=custrev.cumsum()/custrev.sum()
fig,ax=plt.subplots(figsize=(7,4)); ax.plot(np.arange(1,len(share)+1)/len(share),share); ax.plot([0,1],[0,1],"--",color="gray"); ax.set(title="Customer revenue concentration",xlabel="Share of customers (ranked)",ylabel="Cumulative revenue share"); fig.tight_layout(); fig.savefig(figdir/"pareto_curve.png",dpi=150); plt.close(fig)
cutoff=pd.Timestamp("2011-03-01")
# The engine tunes on the earlier 2010 snapshot, then evaluates predictions at the
# chosen 1 Mar 2011 cutoff against actual Mar-Aug 2011 revenue.
customers,bundle,pred=train_engine(tx,cutoff,return_evaluation=True)
metrics=score_models(customers.future_revenue,pred); metrics.to_csv(ROOT/"reports"/"model_comparison.csv",index=False)
best=pred["BG/NBD + Gamma-Gamma"]
decile_lift(customers.future_revenue,best).to_csv(ROOT/"reports"/"decile_lift.csv",index=False)
gains_chart(customers.future_revenue,pred,figdir/"cumulative_gains.png")
permutation_importance_table(bundle["gradient_boosting"],customers[FEATURES],customers.future_revenue).to_csv(ROOT/"reports"/"feature_importance.csv",index=False)
X=customers[FEATURES]; customers["predicted_6m_clv"]=np.maximum(bundle["regressor"].predict(customers),0)
classes=list(bundle["churn"][-1].classes_); customers["churn_probability"]=bundle["churn"].predict_proba(X)[:,classes.index(1)] if 1 in classes else 0.
customers["segment"]=assign_segments(customers.predicted_6m_clv,customers.churn_probability)
customers[["Customer ID","predicted_6m_clv","churn_probability","segment"]].to_csv(OUT/"customer_predictions.csv",index=False)
seg=customers.groupby("segment").agg(customers=("Customer ID","count"),predicted_revenue=("predicted_6m_clv","sum"),avg_churn=("churn_probability","mean"))
seg["revenue_share"]=seg.predicted_revenue/seg.predicted_revenue.sum(); seg.to_csv(ROOT/"reports"/"segment_summary.csv")
actions={"Champions":"VIP service, early access and a replenishment reminder; avoid routine discounts.",
         "Loyal":"Personalized cross-sell and relevant replenishment messages.",
         "Potential":"Low-cost onboarding and second-purchase nudges.",
         "At-Risk":"Personalized win-back offer, prioritized by value and tested against a holdout.",
         "Hibernating":"Capped low-cost reactivation test; suppress non-responders."}
seg.reset_index().assign(recommended_action=lambda x:x.segment.map(actions)).to_csv(ROOT/"reports"/"segment_actions.csv",index=False)
# Illustrative planning scenario; these are assumptions, not measured campaign effects.
# At-risk plus Champions are selected for a small pilot. Assumptions: 30% contribution margin,
# 10% incremental retained revenue, and £1 contact cost per selected customer.
selected=customers[customers.segment.isin(["Champions","At-Risk"])]
scenario={"target_segments":"Champions + At-Risk","customers":len(selected),
          "forecast_revenue":selected.predicted_6m_clv.sum(),"assumed_margin":.30,
          "assumed_incremental_retention":.10,"assumed_contact_cost_gbp":1.0}
scenario["estimated_incremental_profit_gbp"]=scenario["forecast_revenue"]*scenario["assumed_margin"]*scenario["assumed_incremental_retention"]
scenario["estimated_campaign_cost_gbp"]=scenario["customers"]*scenario["assumed_contact_cost_gbp"]
scenario["estimated_net_benefit_gbp"]=scenario["estimated_incremental_profit_gbp"]-scenario["estimated_campaign_cost_gbp"]
scenario["estimated_roi_percent"]=100*scenario["estimated_net_benefit_gbp"]/max(scenario["estimated_campaign_cost_gbp"],1)
pd.DataFrame([scenario]).to_csv(ROOT/"reports"/"roi_scenario.csv",index=False)
print("Cleaning audit:\n",audit.to_string(index=False)); print("\nModel comparison:\n",metrics.to_string(index=False)); print("\nSegments:\n",seg.to_string())
