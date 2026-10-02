import streamlit as st

def render(_d,mode="lineage"):
    if mode=="ml":
        st.header("ML Analytics")
        st.caption("CASE PRIORITIZATION · XGBoost ranks cases already created by the scenario engine and 72h case policy. It does not generate transaction alerts.")
        a,b,c,d=st.columns(4)
        a.metric("ROC-AUC","0.877"); b.metric("PR-AUC","0.337"); c.metric("PR-AUC lift","10.43x"); d.metric("Top-100 precision","90.0%")
        st.info("Frozen HOLDOUT evaluation. ML output is a prioritization score for human investigation, not an AML determination.")
        return
    st.header("Governance & Lineage")
    st.caption("Frozen end-to-end decision lineage")
    st.graphviz_chart("""digraph {
      rankdir=LR;
      node [style=filled, fillcolor="#111827", fontcolor="white", color="#475569"];
      Raw[label="Raw Transactions"]; Norm[label="ISO-2 + Features"];
      Rules[label="Scenario Engine\n6 Frozen Rules"]; Alerts[label="Transaction Alerts"];
      Cases[label="Case Management\n72h Policy"]; Features[label="Case Features"];
      ML[label="ML Prioritization\nFrozen XGBoost"]; Queue[label="Investigator Queue"]; Human[label="Human Review"];
      Raw->Norm->Rules->Alerts->Cases->Features->ML->Queue->Human;
    }""")
    st.caption("Detection and prioritization are deliberately separated: scenarios create alerts; ML ranks resulting cases.")
