import streamlit as st
from dashboard.data import load_dashboard
from dashboard import overview,queue,case_view,scenario_lab,lineage_view,model_comparison

st.set_page_config(page_title="AML Investigation Console",page_icon="🔎",layout="wide")
st.markdown("""<style>
.stApp{background:#070b12;color:#f1f5f9}
.block-container{padding-top:1.35rem;max-width:1500px}
[data-testid="stSidebar"]{background:#0b1220;border-right:1px solid #1f2937}
[data-testid="stMetric"]{background:#0f172a;border:1px solid #263244;padding:14px 16px;border-radius:12px}
[data-testid="stMetricLabel"]{color:#94a3b8}
[data-testid="stDataFrame"]{border:1px solid #1f2937;border-radius:10px}
h1,h2,h3{letter-spacing:-.02em;color:#f8fafc}
.stCaption{color:#cbd5e1!important}

/* Metric cards: Streamlit nests labels/values in several wrappers. */
[data-testid="stMetric"] *{color:#f8fafc!important}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] *{color:#cbd5e1!important}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
div[data-testid="stMetricValue"]{color:#ffffff!important}

/* General dark-theme readability */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] *{color:#e5edf7}

</style>""",unsafe_allow_html=True)

d=load_dashboard()
st.title("AML Investigation Console")
st.caption("Detection Strategy  →  Alert Generation  →  Case Management  →  Human Investigation")
st.sidebar.markdown("### Investigation strategy")
strategy=st.sidebar.radio("Prioritization approach",["Scenario-based","Machine Learning"],horizontal=False,
    help="Compare the frozen Scenario-Based V3 detector with the frozen transaction-level ML Detection V1 engine.")
d["strategy"]=strategy
if strategy=="Machine Learning" and d.get("ml_scores") is None:
    st.sidebar.warning("ML scores not built yet. Run the ML feature + scoring pipeline.")
else:
    st.sidebar.caption("Frozen V3 detection · "+("transparent operational score" if strategy=="Scenario-based" else "frozen DEVELOPMENT-trained XGBoost ranking"))
page=st.sidebar.radio("Workspace",["Monitoring Overview","Model Comparison","Scenario Monitoring","Investigator Queue","Case Investigation","ML Analytics","Simulation Lab","Governance & Lineage"])

if page=="Monitoring Overview": overview.render(d)
elif page=="Model Comparison": model_comparison.render(d)
elif page=="Scenario Monitoring": scenario_lab.render(d,"monitor")
elif page=="Investigator Queue": queue.render(d)
elif page=="Case Investigation": case_view.render(d)
elif page=="ML Analytics": lineage_view.render(d,"ml")
elif page=="Simulation Lab": scenario_lab.render(d,"simulate")
else: lineage_view.render(d,"lineage")
