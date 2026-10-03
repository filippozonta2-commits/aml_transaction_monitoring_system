import pandas as pd
import streamlit as st

def render(d,mode="lineage"):
    if mode=="ml":
        st.header("ML Prioritization Analytics")
        st.caption("Frozen DEVELOPMENT-selected XGBoost · alternative downstream ranking of the same V3/V1 cases")
        ml=d.get("ml_scores"); imp=d.get("ml_importance")
        if ml is None:
            st.warning("ML outputs are not built yet. Run build_ml_case_features_v3.py and score_frozen_ml_case_model_v3.py.")
            return
        q=d["queue"].merge(ml,on=["case_id","subject_id"],how="inner",validate="one_to_one")
        a,b,c,e=st.columns(4)
        a.metric("Cases scored",f"{len(q):,}"); b.metric("Model","XGBoost"); c.metric("Training","DEVELOPMENT"); e.metric("HOLDOUT labels","NOT USED")
        st.info("Frozen V3 still generates the alerts. ML only changes which existing cases an investigator sees first.")
        q["rank_shift"]=q["queue_rank"]-q["ml_rank"]
        q["abs_rank_shift"]=q.rank_shift.abs()
        left,right=st.columns(2)
        with left:
            st.subheader("ML score distribution")
            bins=(q.ml_score//5*5).astype(int).astype(str)+"–"+((q.ml_score//5*5)+5).astype(int).astype(str)
            dist=bins.value_counts().rename_axis("ML score").reset_index(name="cases").sort_values("ML score")
            st.bar_chart(dist,x="ML score",y="cases")
        with right:
            st.subheader("Scenario rank vs ML rank")
            st.scatter_chart(q,x="queue_rank",y="ml_rank",size="alert_count")
        st.subheader("Largest ranking changes")
        show=q.sort_values("abs_rank_shift",ascending=False)[["case_id","subject_id","queue_rank","ml_rank","rank_shift","risk_score","ml_score","queue_priority","scenario_count","alert_count","transaction_count","scenarios"]]
        st.dataframe(show.head(100),width="stretch",hide_index=True)
        if imp is not None and len(imp):
            st.subheader("Global XGBoost feature importance")
            st.bar_chart(imp.head(15).sort_values("importance").set_index("feature")["importance"],horizontal=True)
        st.subheader("Model governance")
        st.markdown("**Frozen model:** XGBoost · **Training:** DEVELOPMENT only · **Role:** ranking/prioritization, not detection or AML determination.")
        st.caption("Account-purged DEVELOPMENT validation reference: ROC-AUC 0.9048 · PR-AUC 0.4410 · Precision@250 34.8% · Recall@250 54.4%.")
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
