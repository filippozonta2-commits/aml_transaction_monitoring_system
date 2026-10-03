import streamlit as st

def render(d):
    m=d["metrics"]; strategy=d.get("strategy","Scenario-based")
    p=m[m.scenario.eq("PORTFOLIO_V3")].iloc[0]; ml=d.get("ml_detection")
    st.header("Executive Overview")
    if strategy=="Machine Learning" and ml is not None:
        n=len(ml); alerts=int(ml.ML_ALERT_V1.sum()); positives=int(ml.Is_laundering.sum())
        hits=int(ml.loc[ml.ML_ALERT_V1.eq(1),"Is_laundering"].sum())
        precision=hits/alerts; recall=hits/positives
        title="Frozen ML Detection V1 · XGBoost · threshold 0.653366"
    else:
        n=1898009; alerts=int(p.alerts); hits=int(p.aml_hits); positives=2136
        precision=float(p.precision); recall=float(p.recall)
        title="Frozen Rule-Based Detection V3 · 11 scenarios"
    st.caption(title)
    cols=st.columns(6)
    vals=[("Transactions",f"{n:,}"),("Alerts",f"{alerts:,}"),("Alert rate",f"{alerts/n:.2%}"),
          ("AML captured",f"{hits:,}/{positives:,}"),("Precision",f"{precision:.2%}"),("Recall",f"{recall:.2%}")]
    for x,(name,val) in zip(cols,vals): x.metric(name,val)
    st.caption("Identical KPI definitions in both modes so the two frozen detection engines are directly comparable.")
    if strategy=="Machine Learning" and ml is not None:
        left,right=st.columns(2)
        with left:
            st.subheader("ML probability distribution")
            z=ml.copy(); z["band"]=(z.ml_probability*10).astype(int).clip(0,9)
            dist=z.band.value_counts().sort_index(); dist.index=[f"{i/10:.1f}–{(i+1)/10:.1f}" for i in dist.index]
            st.bar_chart(dist)
        with right:
            st.subheader("Frozen operating point")
            st.metric("Threshold","0.653366"); st.metric("Workload reduction vs Rule V3",f"{1-alerts/int(p.alerts):.2%}")
    else:
        left,right=st.columns(2)
        with left:
            st.subheader("Operational alerts by scenario")
            st.bar_chart(d["scenarios"].set_index("scenario_id")["aggregated_alerts"].sort_values(),horizontal=True)
        with right:
            st.subheader("Rule-based scenario portfolio")
            st.metric("Scenarios","11"); st.metric("Aggregated investigation alerts",f"{len(d['alerts']):,}")
