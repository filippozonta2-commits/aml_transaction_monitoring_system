import streamlit as st
def render(_d,mode="lineage"):
    if mode=="ml":
        st.header("Priority Analytics"); st.caption("Operational Layer V1 · transparent rule-based prioritization")
        st.info("The previous ML prioritization view is retired for V1. risk_score uses scenario severity, multi-scenario coverage, alert density and transaction volume. AML outcome labels are not used.")
        st.markdown("**Bands:** LOW 0–24 · MEDIUM 25–49 · HIGH 50–100. Policy-linked cases have a MEDIUM floor.")
        return
    st.header("Governance & Lineage"); st.caption("Frozen V3 detection → Frozen V1 operational workflow")
    st.graphviz_chart("""digraph {
      rankdir=LR;
      node [style=filled, fillcolor="#111827", fontcolor="white", color="#475569"];
      Raw[label="HOLDOUT Transactions"]; Rules[label="Frozen V3\n11 Scenarios"];
      Tx[label="127,981\nFlagged Transactions"]; Agg[label="39,921\nAggregated Alerts"];
      Cases[label="24,804\nCases"]; Score[label="Rule-based\nPriority Score"];
      Queue[label="HIGH / MEDIUM / LOW"]; Human[label="Human Review"];
      Raw->Rules->Tx->Agg->Cases->Score->Queue->Human;
    }""")
    st.caption("HOLDOUT labels are validation-only and are not used by aggregation, case creation or prioritization.")
