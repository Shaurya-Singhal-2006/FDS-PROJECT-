"""Leakage-safe customer features and six-month targets."""
import numpy as np
import pandas as pd

FEATURES = ["recency_days","frequency","monetary","tenure_days","avg_order_value",
            "interval_mean_days","interval_std_days","distinct_products","return_rate","q4_order_share"]

def build_customer_table(transactions, cutoff, horizon_months=6, returns=None):
    """Features use dates < cutoff; target revenue is from cutoff through horizon."""
    df = transactions.copy(); df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    cutoff = pd.Timestamp(cutoff); end = cutoff + pd.DateOffset(months=horizon_months)
    obs = df[df.InvoiceDate < cutoff].copy()
    future = df[(df.InvoiceDate >= cutoff) & (df.InvoiceDate < end)].copy()
    # Return rate is the share of customer invoices marked as returns before cutoff.
    ret_rate = {}
    if returns is not None and len(returns):
        r=returns.copy(); r["InvoiceDate"]=pd.to_datetime(r["InvoiceDate"])
        r=r[r.InvoiceDate<cutoff]
        nret=r.groupby("Customer ID").Invoice.nunique()
        nbuy=obs.groupby("Customer ID").Invoice.nunique()
        ret_rate=(nret / nret.add(nbuy,fill_value=0)).fillna(0).to_dict()
    rows=[]
    for cid, g in obs.groupby("Customer ID"):
        dates = g.groupby("Invoice").InvoiceDate.min().sort_values()
        orders = len(dates); last = dates.max()
        diffs = dates.diff().dt.total_seconds().dropna()/86400
        total = float(g.Revenue.sum())
        rows.append({"Customer ID":cid,"recency_days":(cutoff-last).days,"frequency":orders,
            "monetary":total,"tenure_days":max((last-dates.min()).days,0),
            "avg_order_value":total/max(orders,1),"interval_mean_days":float(diffs.mean()) if len(diffs) else 0.,
            "interval_std_days":float(diffs.std()) if len(diffs)>1 else 0.,
            "distinct_products":int(g.StockCode.nunique()),"return_rate":float(ret_rate.get(cid,0)),
            "q4_order_share":float((dates.index.to_series().map(lambda x: pd.Timestamp(x).month if False else 0).mean()) if False else
                                  g.loc[g.InvoiceDate.dt.quarter.eq(4),"Invoice"].nunique()/max(orders,1))})
    feat = pd.DataFrame(rows)
    early = df[(df.InvoiceDate >= cutoff) & (df.InvoiceDate < cutoff + pd.Timedelta(days=90))]
    if future.empty: feat["future_revenue"] = 0.; feat["inactive_90d"] = 1
    else:
        y = future.groupby("Customer ID").agg(future_revenue=("Revenue","sum"))
        feat = feat.merge(y, how="left", left_on="Customer ID", right_index=True)
        feat["future_revenue"] = feat.future_revenue.fillna(0.)
        early_customers=set(early["Customer ID"].unique())
        feat["inactive_90d"] = (~feat["Customer ID"].isin(early_customers)).astype(int)
    return feat
