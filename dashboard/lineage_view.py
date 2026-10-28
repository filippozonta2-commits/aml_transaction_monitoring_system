import pandas as pd
import streamlit as st

def render(d,mode="lineage"):
    if mode=="ml":
        st.header("ML Detection Analytics")
        st.caption("Frozen transaction-level XGBoost Detection V1 · independent alternative to Rule-Based V3")
        ml=d.get("ml_detection"); ops=d.get("ml_dev_ops"); ov=d.get("overlap")
        if ml is None:
            st.warning("Frozen ML detector output is unavailable.")
            return
        n=len(ml); alerts=int(ml.ML_ALERT_V1.sum()); positives=int(ml.Is_laundering.sum())
        hits=int(ml.loc[ml.ML_ALERT_V1.eq(1),"Is_laundering"].sum())
        a,b,c1,e=st.columns(4)
        a.metric("Threshold","0.653366"); b.metric("HOLDOUT alerts",f"{alerts:,}")
        c1.metric("Precision",f"{hits/alerts:.2%}"); e.metric("Recall",f"{hits/positives:.2%}")
        left,right=st.columns(2)
        with left:
            st.subheader("Probability distribution")
            z=ml.copy(); z["score_band"]=(z.ml_probability*10).astype(int).clip(0,9)
            dist=z.score_band.value_counts().sort_index()
            dist.index=[f"{i/10:.1f}–{(i+1)/10:.1f}" for i in dist.index]
            st.bar_chart(dist)
        with right:
            st.subheader("DEVELOPMENT operating points")
            if ops is not None:
                show=ops[["alert_rate","alerts","precision","recall","threshold"]].copy()
                st.dataframe(show,width="stretch",hide_index=True)
        if ov is not None:
            st.subheader("Frozen ML V1 vs Rule V3 AML overlap")
            st.bar_chart(ov.set_index("group")["aml_positives"])
        st.subheader("Model governance")
        st.markdown("**Model:** XGBoost · **Training:** full TRAIN · **model/threshold selection:** DEVELOPMENT · **threshold:** 0.653366 · **HOLDOUT:** evaluation only.")
        st.info("ML V1 is a transaction-level detection engine. It is not the older case-prioritization model.")
        return
    st.header("Governance & Lineage"); st.caption("Frozen V3 detection → V1 case workflow → selectable prioritization")
    st.graphviz_chart("""digraph {
      rankdir=LR;
      node [style=filled, fillcolor="#111827", fontcolor="white", color="#475569"];
      Raw[label="HOLDOUT Transactions"]; Rules[label="Frozen V3\\n11 Scenarios"];
      Tx[label="127,981\\nFlagged Transactions"]; Agg[label="39,921\\nAggregated Alerts"];
      Cases[label="24,804\\nCases"]; Scenario[label="Scenario-based\\nPriority"];
      ML[label="Frozen XGBoost\\nPriority"]; Human[label="Human Review"];
      Raw->Rules->Tx->Agg->Cases; Cases->Scenario->Human; Cases->ML->Human;
    }""")
    st.caption("Both paths use the same Frozen V3 detections and V1 cases. HOLDOUT labels are not used for operational prioritization.")
