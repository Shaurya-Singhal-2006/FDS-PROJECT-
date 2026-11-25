"""Fit simple, probabilistic and machine-learning CLV models."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features import FEATURES

class HistoricalAverageModel:
    """Fallback predictor that spreads observed revenue evenly over six months."""
    def predict(self, frame):
        return np.maximum(frame.monetary.to_numpy()*6/frame.observation_months.clip(lower=1).to_numpy(),0)

class BGNGammaGammaModel:
    """Lifetimes BG/NBD purchase model multiplied by Gamma-Gamma order value."""
    def __init__(self, penalizer=0.01):
        self.penalizer=penalizer

    def fit(self, train):
        data=train.copy()
        data["T"]=(data.recency_days+data.tenure_days).clip(lower=1)/30.4375
        data["recency_life"]=data.tenure_days.clip(lower=0)/30.4375
        data["frequency_life"]=(data.frequency-1).clip(lower=0)
        repeat=data.frequency_life>0
        self.default_spend=float(data.loc[repeat,"avg_order_value"].mean())
        # Lifetimes fitter objects contain local lambdas and cannot be pickled.
        # Store just their compact calibration columns and recreate fitters after load.
        self.calibration=data.loc[:,["frequency_life","recency_life","T","avg_order_value"]].copy()
        self._fitters()
        return self

    def _fitters(self):
        from lifetimes import BetaGeoFitter, GammaGammaFitter
        d=self.calibration; repeat=d.frequency_life>0
        self.bg=BetaGeoFitter(penalizer_coef=self.penalizer).fit(d.frequency_life,d.recency_life,d["T"])
        self.gg=GammaGammaFitter(penalizer_coef=self.penalizer).fit(d.loc[repeat,"frequency_life"],d.loc[repeat,"avg_order_value"])

    def __getstate__(self):
        """Exclude unpicklable Lifetimes objects; keep data needed to recreate them."""
        state=self.__dict__.copy(); state.pop("bg",None); state.pop("gg",None)
        return state

    def __setstate__(self,state):
        self.__dict__.update(state); self._fitters()

    def predict(self, frame):
        data=frame.copy()
        data["T"]=(data.recency_days+data.tenure_days).clip(lower=1)/30.4375
        data["rl"]=data.tenure_days.clip(lower=0)/30.4375
        data["fl"]=(data.frequency-1).clip(lower=0)
        purchases=self.bg.conditional_expected_number_of_purchases_up_to_time(6,data.fl,data.rl,data["T"])
        spend=pd.Series(self.default_spend,index=data.index,dtype=float)
        repeat=data.fl>0
        if repeat.any():
            spend.loc[repeat]=self.gg.conditional_expected_average_profit(data.loc[repeat,"fl"],data.loc[repeat,"avg_order_value"])
        spend=spend.replace([np.inf,-np.inf],np.nan).fillna(self.default_spend).clip(lower=0)
        purchases=pd.Series(purchases,index=data.index).replace([np.inf,-np.inf],np.nan).fillna(0).clip(lower=0)
        return (purchases*spend).to_numpy()

def fit_predict(train, valid, seed=42, tuning_data=None):
    """Fit methods on earlier snapshots and evaluate on a later time snapshot.

    Optionally choose tree regularization on a separate earlier chronological
    validation snapshot, then refit on the complete eligible training panel.
    """
    train = train.dropna(subset=["future_revenue"]); valid = valid.copy()
    X, y = train[FEATURES], train.future_revenue
    # Historical average is the transparent baseline: observed revenue spread evenly over six months.
    baseline = valid.monetary * (6 / valid.observation_months.clip(lower=1))
    pred = {"Historical average": baseline.clip(lower=0).to_numpy()}
    # BG/NBD + Gamma-Gamma model; one-time buyers use the repeat-buyer mean basket value.
    try:
        probabilistic=BGNGammaGammaModel().fit(train)
        pred["BG/NBD + Gamma-Gamma"] = probabilistic.predict(valid)
    except Exception as exc:
        # Keep the pipeline usable where lifetimes cannot fit degenerate data, but make the reason visible.
        print(f"BG/NBD + Gamma-Gamma unavailable; using historical-average fallback: {exc}")
        probabilistic=HistoricalAverageModel()
        pred["BG/NBD + Gamma-Gamma"] = baseline.clip(lower=0).to_numpy()
    gb_params={"max_iter":120,"learning_rate":0.06,"max_leaf_nodes":15,"l2_regularization":1.0,"random_state":seed}
    if tuning_data is not None:
        tune_train,tune_valid=tuning_data
        choices=[]
        for leaves in (7,15):
            for l2 in (1.0,10.0):
                candidate=make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingRegressor(
                    max_iter=120,learning_rate=0.06,max_leaf_nodes=leaves,l2_regularization=l2,random_state=seed))
                candidate.fit(tune_train[FEATURES],tune_train.future_revenue)
                err=mean_absolute_error(tune_valid.future_revenue,np.maximum(candidate.predict(tune_valid[FEATURES]),0))
                choices.append((err,leaves,l2))
        _,gb_params["max_leaf_nodes"],gb_params["l2_regularization"]=min(choices)
    ml = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(**gb_params))
    ml.fit(X, y); pred["Gradient boosting"] = np.maximum(ml.predict(valid[FEATURES]), 0)
    # inactive_90d means no purchase transaction in the first 90 days after cutoff.
    churn = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        n_estimators=120, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=-1))
    churn.fit(X, train.inactive_90d.astype(int))
    valid["churn_probability"] = churn.predict_proba(valid[FEATURES])[:, list(churn[-1].classes_).index(1)] if 1 in churn[-1].classes_ else 0.
    return pred, ml, churn, probabilistic

def score_models(actual, predictions):
    """Return regression metrics plus top-20% revenue capture for each model."""
    actual=np.asarray(actual,dtype=float); rows=[]
    for name, p in predictions.items():
        p=np.maximum(np.asarray(p,dtype=float),0); k=max(1,int(np.ceil(len(p)*.2)))
        order=np.argsort(p)[::-1]
        rows.append({"model":name,"MAE":mean_absolute_error(actual,p),"RMSE":mean_squared_error(actual,p)**.5,
                     "R2":r2_score(actual,p),"Top 20% revenue captured":float(actual[order[:k]].sum()/max(actual.sum(),1e-9))})
    return pd.DataFrame(rows)
