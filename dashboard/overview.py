import streamlit as st
def render(d):
    q=d["queue"]; sc=d["scenarios"]; m=d["metrics"]; p=m[m.scenario.eq("PORTFOLIO_V3")].iloc[0]
    st.header("Executive Overview"); st.caption("Frozen Detection V3 · Operational Layer V1")
    cols=st.columns(5)
    vals=[("Transactions","1,898,009"),("Flagged transactions",f"{int(p.alerts):,}"),("Aggregated alerts",f"{len(d['alerts']):,}"),("Cases",f"{len(q):,}"),("HOLDOUT recall",f"{p.recall:.2%}")]
    for x,(n,v) in zip(cols,vals):x.metric(n,v)
    a,b=st.columns(2)
    with a:
        st.subheader("Operational alerts by scenario"); st.bar_chart(sc.set_index("scenario_id")["aggregated_alerts"].sort_values(),horizontal=True)
    with b:
        st.subheader("Investigator queue"); st.bar_chart(q["queue_priority"].value_counts().reindex(["HIGH","MEDIUM","LOW"]).fillna(0))
    st.info("Detection performance is frozen HOLDOUT validation. Aggregation and prioritization are downstream workflow layers.")
