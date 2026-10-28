"""Streamlit investigation dashboard for simulated AML V3 DEVELOPMENT alerts.

Launch: streamlit run src/dashboard_v3.py
"""
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AML Investigation Console", page_icon="🔎", layout="wide")
st.title("AML Transaction Monitoring | Investigation Console")
st.caption("Simulated SAML-D data • V3 DEVELOPMENT alerts • Educational prototype")
st.warning("Priority scores are heuristic, not probabilities. Alerts are not SAR determinations.")

path = Path("results/scenario_v3/investigation_queue.csv")
if not path.exists():
    st.error("Missing investigation queue. Run: python3 -m src.scenario_engine_v3")
    st.stop()

@st.cache_data
def load_queue(path_str, modified):
    return pd.read_csv(path_str, parse_dates=["Transaction_timestamp"])

alerts = load_queue(str(path), path.stat().st_mtime)
required = {"Alert_id", "Priority", "Priority_score", "Triggered_scenarios",
            "Alert_reason", "Sender_account", "Receiver_account", "Amount"}
missing = required.difference(alerts.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

with st.sidebar:
    st.header("Filters")
    priority = st.multiselect("Priority", ["HIGH", "MEDIUM", "LOW"],
                              default=["HIGH", "MEDIUM", "LOW"])
    scenario = st.multiselect("Scenario", ["Smurfing", "Fan_Out", "Structuring", "Fan_In"])
    account = st.text_input("Sender / receiver account contains")
    min_score = st.slider("Minimum priority score", 0, 100, 0)

view = alerts.loc[alerts["Priority"].isin(priority)
                  & alerts["Priority_score"].ge(min_score)].copy()
if scenario:
    view = view.loc[view["Triggered_scenarios"].fillna("").apply(
        lambda s: any(x in s.split(";") for x in scenario))]
if account:
    search = account.strip()
    view = view.loc[
        view["Sender_account"].astype(str).str.contains(search, case=False, regex=False)
        | view["Receiver_account"].astype(str).str.contains(search, case=False, regex=False)
    ]

a, b, c, d = st.columns(4)
a.metric("Total alerts", f"{len(alerts):,}")
b.metric("Filtered alerts", f"{len(view):,}")
c.metric("High priority", f"{(view['Priority'] == 'HIGH').sum():,}")
d.metric("Mean priority score", f"{view['Priority_score'].mean():.1f}" if len(view) else "—")

left, right = st.columns(2)
with left:
    st.subheader("Priority distribution")
    counts = view["Priority"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    st.bar_chart(counts)
with right:
    st.subheader("Scenario triggers")
    names = ["Smurfing", "Fan_Out", "Structuring", "Fan_In"]
    triggers = pd.Series({
        name: view["Triggered_scenarios"].fillna("").apply(
            lambda s: name in s.split(";")).sum() for name in names
    })
    st.bar_chart(triggers)

st.subheader("Investigation queue")
columns = ["Alert_id", "Transaction_timestamp", "Sender_account", "Receiver_account",
           "Amount", "Triggered_scenarios", "Priority_score", "Priority"]
st.dataframe(view[columns], use_container_width=True, hide_index=True)
st.download_button("Download filtered alerts", view.to_csv(index=False),
                   "aml_filtered_investigation_queue.csv", "text/csv")

st.subheader("Alert details")
if view.empty:
    st.info("No alerts match the selected filters.")
else:
    selected_id = st.selectbox("Select alert", view["Alert_id"].tolist())
    selected = view.loc[view["Alert_id"].eq(selected_id)].iloc[0]
    st.write(f"**Triggered scenarios:** {selected['Triggered_scenarios']}")
    st.write(f"**Reason:** {selected['Alert_reason']}")
    st.write(f"**Priority:** {selected['Priority']} ({selected['Priority_score']}/100)")
    st.write(f"**Sender:** {selected['Sender_account']}  |  **Receiver:** {selected['Receiver_account']}")
    st.write(f"**Transaction amount (dataset units):** {selected['Amount']:,.2f}")
    st.caption("An analyst would review customer context and transaction history before disposition.")
