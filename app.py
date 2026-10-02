import streamlit as st
from dashboard.data import load_dashboard
from dashboard import overview,queue,case_view,scenario_lab,lineage_view

st.set_page_config(page_title="AML Investigation Console",page_icon="🔎",layout="wide")
d=load_dashboard()
st.title("AML Investigation Console")
st.caption("Rule-based transaction monitoring · 72h case management · frozen XGBoost prioritization")
page=st.sidebar.radio("Workspace",["Executive Overview","Investigator Queue","Case Investigation","Scenario Lab","Model & Lineage"])
{"Executive Overview":overview.render,"Investigator Queue":queue.render,"Case Investigation":case_view.render,
 "Scenario Lab":scenario_lab.render,"Model & Lineage":lineage_view.render}[page](d)
