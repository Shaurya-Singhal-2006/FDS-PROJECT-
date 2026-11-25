"""Streamlit interface for portfolio CLV predictions."""
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.predict import predict_file

st.set_page_config(page_title="Customer Value Engine",page_icon="📈",layout="wide")
st.markdown("<style>.stApp{background:#f6f8fb}.block-container{padding-top:2rem}.metric{background:white}</style>",unsafe_allow_html=True)
st.title("Customer Lifetime Value Engine")
st.caption("Six-month customer value forecasts and retention priorities · UCI Online Retail II")
pages=["Overview KPIs","Segment explorer","Customer lookup","Model performance","Upload transactions"]
page=st.sidebar.radio("Navigate",pages)
pred_path=ROOT/"data"/"processed"/"customer_predictions.csv"
if pred_path.exists(): df=pd.read_csv(pred_path)
else: df=pd.DataFrame()
if page=="Overview KPIs":
    st.subheader("Portfolio overview")
    if df.empty: st.info("Run the pipeline to create data/processed/customer_predictions.csv.")
    else:
        a,b,c=st.columns(3); a.metric("Customers",f"{len(df):,}"); b.metric("Predicted 6-month value",f"£{df.predicted_6m_clv.sum():,.0f}"); c.metric("High-risk customers",f"{(df.churn_probability>=.5).sum():,}")
        st.bar_chart(df.groupby("segment").predicted_6m_clv.sum())
elif page=="Segment explorer":
    st.subheader("Explore segments")
    if df.empty: st.info("Run the pipeline first.")
    else:
        g=df.groupby("segment").agg(customers=("Customer ID","count"),predicted_revenue=("predicted_6m_clv","sum"),mean_churn=("churn_probability","mean"))
        actions={"Champions":"VIP service, early access and replenishment reminder","Loyal":"Personalized cross-sell and replenishment","Potential":"Second-purchase onboarding nudge","At-Risk":"Test a personalized win-back offer","Hibernating":"Capped reactivation test; suppress non-responders"}
        g["recommended_action"]=[actions.get(s,"") for s in g.index]
        g["share_of_revenue"]=g.predicted_revenue/g.predicted_revenue.sum(); st.dataframe(g.style.format({"predicted_revenue":"£{:,.0f}","mean_churn":"{:.1%}","share_of_revenue":"{:.1%}"}),use_container_width=True)
        st.dataframe(df[df.segment==st.selectbox("Segment",sorted(df.segment.unique()))],use_container_width=True)
elif page=="Customer lookup":
    st.subheader("Customer lookup")
    if df.empty: st.info("Run the pipeline first.")
    else:
        q=st.text_input("Customer ID"); st.dataframe(df[df["Customer ID"].astype(str).str.contains(q,case=False,regex=False)] if q else df.head(0),use_container_width=True)
elif page=="Model performance":
    st.subheader("Out-of-time model evaluation")
    path=ROOT/"reports"/"model_comparison.csv"
    if path.exists(): st.dataframe(pd.read_csv(path),use_container_width=True)
    else: st.info("Run the modeling notebook to generate performance metrics.")
    img=ROOT/"reports"/"figures"/"cumulative_gains.png"
    if img.exists(): st.image(str(img),use_container_width=True)
elif page=="Upload transactions":
    st.subheader("Score a transaction file")
    uploaded=st.file_uploader("CSV or Excel",type=["csv","xlsx"]); cutoff=st.date_input("Feature cutoff date",value=pd.Timestamp("2011-03-01").date())
    if uploaded and st.button("Generate predictions"):
        temp=ROOT/"data"/"raw"/f"upload_{uploaded.name}"; temp.write_bytes(uploaded.getvalue())
        result=predict_file(temp,cutoff); st.dataframe(result,use_container_width=True); st.download_button("Download predictions",result.to_csv(index=False),"customer_predictions.csv","text/csv")
