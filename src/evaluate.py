"""Evaluation helpers and portfolio figures."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def decile_lift(actual, predicted):
    """Revenue by predicted-value decile (10 is highest)."""
    d=pd.DataFrame({"actual":actual,"predicted":predicted})
    d["decile"]=pd.qcut(d.predicted.rank(method="first"),10,labels=False)+1
    out=d.groupby("decile",observed=False).actual.agg(["count","sum","mean"]).reset_index()
    out["lift_vs_average"]=out["mean"]/max(d.actual.mean(),1e-9)
    return out.sort_values("decile",ascending=False)

def gains_chart(actual, predicted, path):
    series=predicted if isinstance(predicted,dict) else {"Model ranking":predicted}
    fig,ax=plt.subplots(figsize=(7,5))
    for label, values in series.items():
        order=np.argsort(np.asarray(values))[::-1]; y=np.asarray(actual)[order]
        cum=np.cumsum(y)/max(y.sum(),1e-9); pct=np.arange(1,len(y)+1)/len(y)
        ax.plot(pct,cum,label=label,lw=2)
    ax.plot([0,1],[0,1],"--",color="gray",label="Random selection")
    ax.set(xlabel="Share of customers targeted",ylabel="Share of future revenue captured",title="Cumulative gains"); ax.legend(); fig.tight_layout()
    Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160); plt.close(fig)

def permutation_importance_table(model, X, y, seed=42):
    from sklearn.inspection import permutation_importance
    r=permutation_importance(model,X,y,n_repeats=5,random_state=seed,scoring="neg_mean_absolute_error")
    return pd.DataFrame({"feature":X.columns,"importance":r.importances_mean}).sort_values("importance",ascending=False)
