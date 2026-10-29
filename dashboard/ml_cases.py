"""Independent ML V1 operational case queue and case investigation views."""
import pandas as pd
import streamlit as st

def render(d):
    cases = d.get("ml_cases")
    alerts = d.get("ml_alerts")
    links = d.get("ml_case_alert")
    evidence = d.get("ml_alert_tx")
    st.header("ML Case Management")
    st.caption("Frozen XGBoost Detection V1 → account-session aggregation → independent ML cases. No AML labels used in case generation.")
    if any(x is None for x in (cases, alerts, links, evidence)):
        st.warning("ML operational case artifacts have not been generated. Run python3 -m src.build_ml_aggregated_alert_layer_v1 and then python3 -m src.build_ml_case_management_v1.")
        return
    if cases.empty:
        st.info("No ML investigation cases.")
        return
    a,b,c = st.columns(3)
    a.metric("ML investigation cases", f"{len(cases):,}")
    b.metric("Aggregated ML alerts", f"{len(alerts):,}")
    c.metric("Linked unique transactions", f"{evidence.transaction_id.nunique():,}")
    tab_queue, tab_case = st.tabs(["ML Investigator Queue", "ML Case Investigation"])
    with tab_queue:
        n = st.selectbox("Cases to display", [100,250,500,1000], index=1)
        columns = [x for x in ["queue_rank","case_id","subject_id","ml_probability_max","ml_probability_mean","alert_count","transaction_count","case_created_at","last_alert_at","case_status"] if x in cases.columns]
        st.dataframe(cases.sort_values("queue_rank")[columns].head(n), hide_index=True, use_container_width=True)
        st.caption("Priority is relative ML ranking; it is not a SAR recommendation or calibrated case-level probability.")
    with tab_case:
        cid = st.selectbox("Select ML case", cases.sort_values("queue_rank").case_id.tolist())
        row = cases.loc[cases.case_id.eq(cid)].iloc[0]
        st.subheader(f"{cid} · Account {row.subject_id}")
        a,b,c = st.columns(3)
        a.metric("Rank", f"#{int(row.queue_rank):,}")
        b.metric("Maximum ML probability", f"{row.ml_probability_max:.2%}")
        c.metric("Unique transactions", f"{int(row.transaction_count):,}")
        ids = links.loc[links.case_id.eq(cid),"alert_id"]
        selected_alerts = alerts[alerts.alert_id.isin(ids)]
        selected_evidence = evidence[evidence.alert_id.isin(ids)]
        st.subheader("Aggregated alerts")
        st.dataframe(selected_alerts, hide_index=True, use_container_width=True)
        st.subheader("Transaction evidence and lineage")
        st.dataframe(selected_evidence, hide_index=True, use_container_width=True)
        st.caption("The frozen transaction-level ML detector and HOLDOUT metrics remain unchanged.")
