"""Small deterministic guardrails for leakage and non-negative scores."""
import pandas as pd
import numpy as np
from src.features import build_customer_table, FEATURES
from src.models import fit_predict, BGNGammaGammaModel
import joblib

def sample_transactions():
    return pd.DataFrame({"Invoice":["a","b","c"],"StockCode":["x","x","y"],
        "InvoiceDate":pd.to_datetime(["2020-01-01","2020-02-01","2020-08-01"]),
        "Quantity":[1,2,1],"Price":[10.,10.,100.],"Customer ID":["1","1","1"],"Revenue":[10.,20.,100.]})

def test_future_rows_do_not_change_features():
    df=sample_transactions(); cutoff="2020-06-01"
    one=build_customer_table(df,cutoff)
    changed=df.copy(); changed.loc[changed.InvoiceDate>=cutoff,"Revenue"]*=99
    changed.loc[changed.InvoiceDate>=cutoff,"Price"]*=99
    two=build_customer_table(changed,cutoff)
    pd.testing.assert_frame_equal(one[FEATURES],two[FEATURES])
    assert one.future_revenue.iloc[0] != two.future_revenue.iloc[0]

def test_model_predictions_are_nonnegative():
    rows=[]
    for i in range(30):
        r={f:float(i+1) for f in FEATURES}; r.update(frequency=i%5+1,tenure_days=0. if i%5==0 else float(i+1),
            future_revenue=float(i*5),inactive_90d=int(i%2),observation_months=12)
        rows.append(r)
    train=pd.DataFrame(rows); valid=train.iloc[:8].copy()
    predictions,_,_,_=fit_predict(train,valid)
    assert all(np.all(np.asarray(p)>=0) for p in predictions.values())

def test_probabilistic_model_survives_joblib_roundtrip(tmp_path):
    rows=[]
    for i in range(40):
        frequency=i%5+1
        rows.append({"frequency":frequency,"tenure_days":0. if frequency==1 else float(i+1),
            "recency_days":float(10+i),"avg_order_value":float(5+i),"monetary":float(50+i),
            "observation_months":12.})
    frame=pd.DataFrame(rows)
    model=BGNGammaGammaModel().fit(frame); expected=model.predict(frame)
    path=tmp_path/"clv.joblib"; joblib.dump(model,path)
    loaded=joblib.load(path); actual=loaded.predict(frame)
    assert np.allclose(expected,actual)
    assert np.all(actual>=0)
