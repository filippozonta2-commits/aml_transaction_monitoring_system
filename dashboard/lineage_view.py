import streamlit as st

def render(_d):
    st.header("System & ML Lineage")
    st.graphviz_chart('''digraph {
      rankdir=LR;
      Raw[label="Raw Transactions"]; Norm[label="ISO-2 + Features"];
      Rules[label="6 Frozen Scenarios"]; Alerts[label="Transaction Alerts"];
      Cases[label="72h Cases"]; Features[label="20 Case Features"];
      ML[label="Frozen XGBoost"]; Queue[label="Investigator Queue"]; Human[label="Human Review"];
      Raw->Norm->Rules->Alerts->Cases->Features->ML->Queue->Human;
    }''')
    a,b,c,d=st.columns(4)
    a.metric("ROC-AUC","0.877"); b.metric("PR-AUC","0.337"); c.metric("PR-AUC lift","10.43x"); d.metric("Top-100 precision","90.0%")
