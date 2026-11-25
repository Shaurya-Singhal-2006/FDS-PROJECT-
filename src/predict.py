"""Train the selected model and score a new transaction file."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data_prep import load_raw, clean_transactions, OUT, ROOT
from src.features import build_customer_table, FEATURES
from src.models import fit_predict

def train_engine(clean_df, cutoff, model_path=None, return_evaluation=False):
    """Train from rolling historical snapshots and persist a reproducible bundle."""
    cutoff=pd.Timestamp(cutoff)
    return_file=OUT/"transactions_returns.csv"
    returns=pd.read_csv(return_file) if return_file.exists() else None
    snapshots=[]
    for date in pd.date_range("2010-01-01", cutoff-pd.DateOffset(months=6), freq="MS"):
        t=build_customer_table(clean_df,date,returns=returns)
        t["observation_months"]=(date-clean_df.InvoiceDate.min()).days/30.4375
        t["snapshot_date"]=date
        snapshots.append(t)
    if not snapshots: raise ValueError("Not enough history before cutoff for a 6-month target.")
    panel=pd.concat(snapshots,ignore_index=True)
    final=build_customer_table(clean_df,cutoff,returns=returns)
    final["observation_months"]=(cutoff-clean_df.InvoiceDate.min()).days/30.4375
    # Earlier Sep 2010 snapshot tunes tree regularization; its target finishes
    # before the final Mar-Aug 2011 holdout begins.
    tune_date=cutoff-pd.DateOffset(months=6)
    tune_train=panel[panel.snapshot_date<=tune_date-pd.DateOffset(months=6)]
    tune_valid=panel[panel.snapshot_date==tune_date]
    tune=(tune_train,tune_valid) if len(tune_train) and len(tune_valid) else None
    pred,ml,churn,clv_model=fit_predict(panel,final,tuning_data=tune)
    bundle={"regressor":clv_model,"gradient_boosting":ml,"churn":churn,"features":FEATURES,"cutoff":str(cutoff.date()),"seed":42,
            "clv_model_name":"BG/NBD + Gamma-Gamma"}
    model_path=Path(model_path or ROOT/"models"/"clv_model.joblib"); model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,model_path)
    return (final,bundle,pred) if return_evaluation else (final,bundle)

def predict_file(transaction_file, cutoff, output_file=None, model_path=None):
    """Return and save customer ID, predicted six-month CLV, churn risk and segment."""
    tx, _ = clean_transactions(load_raw(transaction_file), output_dir=OUT)
    final,bundle=train_engine(tx,cutoff,model_path)
    X=final[bundle["features"]]
    final["predicted_6m_clv"]=np.maximum(bundle["regressor"].predict(final),0)
    final["churn_probability"]=bundle["churn"].predict_proba(X)[:,list(bundle["churn"][-1].classes_).index(1)]
    final["segment"]=assign_segments(final.predicted_6m_clv,final.churn_probability)
    out=final[["Customer ID","predicted_6m_clv","churn_probability","segment"]]
    output_file=Path(output_file or ROOT/"data"/"processed"/"customer_predictions.csv"); output_file.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(output_file,index=False); return out

def assign_segments(clv, churn):
    """Rule-based segments combining predicted value and 90-day inactivity risk."""
    value=pd.Series(clv); risk=pd.Series(churn,index=value.index); hi=value.quantile(.8); mid=value.quantile(.5)
    return np.select([(value>=hi)&(risk<.5),(value>=mid)&(risk<.5),(value>=hi)&(risk>=.5),risk>=.7],
                     ["Champions","Loyal","At-Risk","Hibernating"],default="Potential")

if __name__ == "__main__":
    tx,_=clean_transactions(load_raw())
    final,bundle=train_engine(tx,"2011-03-01")
