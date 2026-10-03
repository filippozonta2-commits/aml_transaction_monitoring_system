import streamlit as st

def render(d):
    q=d["queue"]; sc=d["scenarios"]; m=d["metrics"]
    strategy=d.get("strategy","Scenario-based")
    p=m[m.scenario.eq("PORTFOLIO_V3")].iloc[0]
    ml=d.get("ml_detection")

    st.header("Executive Overview")

    if strategy=="Machine Learning" and ml is not None:
        n=len(ml)
        alerts=int(ml["ML_ALERT_V1"].sum())
        positives=int(ml["Is_laundering"].sum())
        hits=int(ml.loc[ml["ML_ALERT_V1"].eq(1),"Is_laundering"].sum())
        recall=hits/positives if positives else 0
        precision=hits/alerts if alerts else 0
        st.caption("Frozen ML Detection V1 · XGBoost · threshold 0.653366")
        cols=st.columns(5)
        vals=[
            ("Transactions",f"{n:,}"),
            ("ML alerts",f"{alerts:,}"),
            ("Alert rate",f"{alerts/n:.2%}"),
            ("AML captured",f"{hits:,}/{positives:,}"),
            ("HOLDOUT recall",f"{recall:.2%}"),
        ]
        for x,(name,val) in zip(cols,vals): x.metric(name,val)
        a,b=st.columns(2)
        with a:
            st.subheader("ML score distribution")
            scored=ml.copy()
            scored["score_band"]=(scored["ml_probability"]*10).astype(int).clip(0,9)
            dist=scored["score_band"].value_counts().sort_index()
            dist.index=[f"{i/10:.1f}–{(i+1)/10:.1f}" for i in dist.index]
            st.bar_chart(dist)
        with b:
            st.subheader("Detection performance")
            st.metric("Precision",f"{precision:.2%}")
            st.metric("Recall",f"{recall:.2%}")
            st.metric("Workload reduction vs Rule V3",f"{1-alerts/int(p.alerts):.2%}")
        st.info("ML Detection V1 is an independent frozen transaction-level detector. Case-management metrics shown in the Scenario-Based workflow are not reused here.")
        return

    st.caption("Frozen Rule-Based Detection V3 · Operational Layer V1")
    cols=st.columns(5)
    vals=[
        ("Transactions","1,898,009"),
        ("Flagged transactions",f"{int(p.alerts):,}"),
        ("Aggregated alerts",f"{len(d['alerts']):,}"),
        ("Cases",f"{len(q):,}"),
        ("HOLDOUT recall",f"{p.recall:.2%}"),
    ]
    for x,(name,val) in zip(cols,vals): x.metric(name,val)
    a,b=st.columns(2)
    with a:
        st.subheader("Operational alerts by scenario")
        st.bar_chart(sc.set_index("scenario_id")["aggregated_alerts"].sort_values(),horizontal=True)
    with b:
        st.subheader("Investigator queue")
        st.bar_chart(q["queue_priority"].value_counts().reindex(["HIGH","MEDIUM","LOW"]).fillna(0))
    st.info("Detection performance is frozen HOLDOUT validation. Aggregation and prioritization are downstream workflow layers.")
