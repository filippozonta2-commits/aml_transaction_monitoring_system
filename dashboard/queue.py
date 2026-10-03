import streamlit as st
def render(d):
    q=d["queue"]; st.header("Prioritized Investigator Queue")
    st.caption("Transparent rule-based Operational Layer V1 prioritization · no AML labels used")
    a,b,c,e=st.columns(4)
    bands=a.multiselect("Priority",["HIGH","MEDIUM","LOW"],default=["HIGH","MEDIUM"])
    policy=b.selectbox("Policy",["All","Policy-linked","Non-policy"])
    coverage=c.selectbox("Coverage",["All","Multi-scenario","Single-scenario"])
    n=e.selectbox("Rows",[100,250,500,1000],index=1)
    z=q[q.queue_priority.isin(bands)].copy()
    if policy=="Policy-linked":z=z[z.policy_flag.astype(bool)]
    elif policy=="Non-policy":z=z[~z.policy_flag.astype(bool)]
    if coverage=="Multi-scenario":z=z[z.scenario_count.gt(1)]
    elif coverage=="Single-scenario":z=z[z.scenario_count.eq(1)]
    cols=["queue_rank","queue_priority","case_id","subject_id","risk_score","alert_count","transaction_count","scenario_count","policy_flag","case_created_at","last_alert_at","scenarios"]
    st.dataframe(z[cols].head(n),use_container_width=True,hide_index=True)
