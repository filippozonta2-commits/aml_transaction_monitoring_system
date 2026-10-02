import pandas as pd
import streamlit as st

def render(d):
    q=d["queue"]; lin=d["lineage"]
    st.header("Case Investigation")
    cid=st.selectbox("Case",q.CASE_ID.tolist())
    r=q[q.CASE_ID.eq(cid)].iloc[0]
    a,b,c,e=st.columns(4)
    a.metric("Priority rank",f"#{int(r.priority_rank):,}")
    b.metric("Risk score",f"{r.risk_score:.3f}")
    c.metric("Alerts",f"{int(r.transaction_alerts):,}")
    e.metric("Alert amount",f"${r.total_alert_amount:,.0f}")
    st.write("**Scenarios:**",r.scenario_list)
    ev=lin[lin.CASE_ID.eq(cid)].copy()
    if len(ev):
        ev["ts"]=pd.to_datetime(ev.ts)
        st.subheader("Alert timeline")
        st.scatter_chart(ev,x="ts",y="holdout_row_id",color="scenario")
        st.dataframe(ev.sort_values("ts"),use_container_width=True,hide_index=True)
    st.caption("The frozen ML score prioritizes human review; it is not an AML determination.")
