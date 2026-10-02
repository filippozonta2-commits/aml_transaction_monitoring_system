import streamlit as st
from dashboard.data import load_dashboard
from dashboard import overview,queue,case_view,scenario_lab,lineage_view

st.set_page_config(page_title="AML Investigation Console",page_icon="🔎",layout="wide")
st.markdown("""<style>
.stApp{background:#070b12;color:#f1f5f9}
.block-container{padding-top:1.35rem;max-width:1500px}
[data-testid="stSidebar"]{background:#0b1220;border-right:1px solid #1f2937}
[data-testid="stMetric"]{background:#0f172a;border:1px solid #263244;padding:14px 16px;border-radius:12px}
[data-testid="stMetricLabel"]{color:#94a3b8}
[data-testid="stDataFrame"]{border:1px solid #1f2937;border-radius:10px}
h1,h2,h3{letter-spacing:-.02em;color:#f8fafc}
.stCaption{color:#94a3b8}
</style>""",unsafe_allow_html=True)

d=load_dashboard()
st.title("AML Investigation Console")
st.caption("Scenario Detection  →  Case Management  →  ML Prioritization  →  Human Investigation")
page=st.sidebar.radio("Workspace",["Monitoring Overview","Scenario Monitoring","Investigator Queue","Case Investigation","ML Analytics","Simulation Lab","Governance & Lineage"])

if page=="Monitoring Overview": overview.render(d)
elif page=="Scenario Monitoring": scenario_lab.render(d,"monitor")
elif page=="Investigator Queue": queue.render(d)
elif page=="Case Investigation": case_view.render(d)
elif page=="ML Analytics": lineage_view.render(d,"ml")
elif page=="Simulation Lab": scenario_lab.render(d,"simulate")
else: lineage_view.render(d,"lineage")
