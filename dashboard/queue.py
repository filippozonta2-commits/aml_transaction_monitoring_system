import streamlit as st
def render(d):
    q=d["queue"].copy()
    strategy=d.get("strategy","Scenario-based")
    st.header("Prioritized Investigator Queue")
    if strategy=="Machine Learning" and d.get("ml_scores") is not None:
        q=q.merge(d["ml_scores"][["case_id","ml_probability","ml_score","ml_rank"]],on="case_id",how="left").sort_values("ml_rank")
        st.caption("Frozen DEVELOPMENT-trained XGBoost ranking; detection and evidence remain Frozen V3.")
    else:
        q=q.sort_values("queue_rank")
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
    cols=(["ml_rank","ml_score","ml_probability"] if strategy=="Machine Learning" and "ml_rank" in z.columns else ["queue_rank"])+["queue_priority","case_id","subject_id","risk_score","alert_count","transaction_count","scenario_count","policy_flag","case_created_at","last_alert_at","scenarios"]
    st.dataframe(z[cols].head(n),width="stretch",hide_index=True)
