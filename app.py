import streamlit as st
from dashboard.data import load_dashboard
from dashboard import overview,queue,case_view,scenario_lab,lineage_view

st.set_page_config(page_title="AML Investigation Console",page_icon="🔎",layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1500px}
[data-testid="stMetric"]{background:#111827;border:1px solid #263244;padding:14px 16px;border-radius:12px}
[data-testid="stMetricLabel"]{color:#94a3b8}
[data-testid="stSidebar"]{border-right:1px solid #263244}
h1,h2,h3{letter-spacing:-.02em}
</style>""",unsafe_allow_html=True)
d=load_dashboard()
st.title("AML Investigation Console")
st.caption("Transaction Monitoring  /  Case Management  /  ML Prioritization")
page=st.sidebar.radio("Workspace",["Executive Overview","Investigator Queue","Case Investigation","Scenario Lab","Model & Lineage"])
{"Executive Overview":overview.render,"Investigator Queue":queue.render,"Case Investigation":case_view.render,
 "Scenario Lab":scenario_lab.render,"Model & Lineage":lineage_view.render}[page](d)
