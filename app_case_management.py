"""Streamlit Case Management Dashboard for frozen Operational Layer V1.

Read-only presentation layer. It consumes frozen outputs and does not recalculate
detection flags, alert aggregation, case grouping, or priority scores.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

BASE=Path("results/case_management_v3")
QUEUE=BASE/"prioritized/case_priority_queue.csv"
CASE_ALERT=BASE/"aggregated_cases/case_alert.csv"
ALERT=BASE/"aggregated/alert.csv"
ALERT_TX=BASE/"aggregated/alert_transaction.csv"

st.set_page_config(page_title="AML Case Management V3",page_icon="🔎",layout="wide")
st.title("AML Transaction Monitoring — Case Management")
st.caption("Frozen Detection V3 · Operational Layer V1 · Read-only investigator dashboard")

@st.cache_data
def load():
 q=pd.read_csv(QUEUE)
 ca=pd.read_csv(CASE_ALERT)
 al=pd.read_csv(ALERT)
 at=pd.read_csv(ALERT_TX)
 for c in ["case_created_at","last_alert_at"]:
  if c in q: q[c]=pd.to_datetime(q[c],errors="coerce")
 if "alert_created_at" in al: al["alert_created_at"]=pd.to_datetime(al["alert_created_at"],errors="coerce")
 if "last_trigger_at" in al: al["last_trigger_at"]=pd.to_datetime(al["last_trigger_at"],errors="coerce")
 return q,ca,al,at

missing=[str(p) for p in [QUEUE,CASE_ALERT,ALERT,ALERT_TX] if not p.exists()]
if missing:
 st.error("Missing frozen dashboard inputs:\n"+"\n".join(missing))
 st.stop()

q,ca,alerts,at=load()

# Sidebar filters
st.sidebar.header("Queue filters")
priority=st.sidebar.multiselect("Priority",["HIGH","MEDIUM","LOW"],default=["HIGH","MEDIUM","LOW"])
scenario_options=sorted({s for x in q["scenarios"].dropna().astype(str) for s in x.split("|")})
scenario=st.sidebar.multiselect("Scenario",scenario_options)
policy=st.sidebar.selectbox("Policy flag",["All","True","False"])
score_min,score_max=st.sidebar.slider("Risk score",0,100,(0,100),5)

f=q[q.queue_priority.isin(priority) & q.risk_score.between(score_min,score_max)].copy()
if scenario:
 f=f[f.scenarios.fillna("").apply(lambda x:any(s in x.split("|") for s in scenario))]
if policy!="All":
 f=f[f.policy_flag.astype(str).str.lower().eq(policy.lower())]

# KPIs
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Cases",f"{len(f):,}")
c2.metric("HIGH",f"{(f.queue_priority=='HIGH').sum():,}")
c3.metric("MEDIUM",f"{(f.queue_priority=='MEDIUM').sum():,}")
c4.metric("LOW",f"{(f.queue_priority=='LOW').sum():,}")
c5.metric("Policy-linked",f"{f.policy_flag.astype(bool).sum():,}")

st.subheader("Investigator queue")
show_cols=["queue_rank","case_id","subject_id","risk_score","queue_priority","alert_count",
           "transaction_count","scenario_count","policy_flag","case_created_at","last_alert_at","scenarios"]
st.dataframe(f[show_cols].sort_values(["risk_score","queue_rank"],ascending=[False,True]),
             use_container_width=True,hide_index=True)

st.subheader("Queue composition")
a,b=st.columns(2)
with a:
 st.caption("Cases by priority")
 st.bar_chart(f["queue_priority"].value_counts().reindex(["HIGH","MEDIUM","LOW"]).fillna(0))
with b:
 st.caption("Risk score distribution")
 bins=pd.cut(f.risk_score,bins=list(range(0,105,5)),include_lowest=True)
 st.bar_chart(bins.value_counts().sort_index())

st.divider()
st.subheader("Case drill-down")
case_ids=f.sort_values("queue_rank").case_id.tolist()
if not case_ids:
 st.info("No cases match the current filters.")
 st.stop()
selected=st.selectbox("Select case",case_ids)
case=q[q.case_id.eq(selected)].iloc[0]

k1,k2,k3,k4,k5=st.columns(5)
k1.metric("Risk score",int(case.risk_score))
k2.metric("Priority",case.queue_priority)
k3.metric("Alerts",int(case.alert_count))
k4.metric("Transactions",int(case.transaction_count))
k5.metric("Scenarios",int(case.scenario_count))

st.write(f"**Case:** {case.case_id}  |  **Subject account:** {case.subject_id}  |  **Policy flag:** {bool(case.policy_flag)}")
st.write(f"**Scenarios:** {case.scenarios}")
st.write(f"**Priority rationale:** {case.priority_reason}")

case_alert_ids=ca.loc[ca.case_id.eq(selected),"alert_id"]
ad=alerts[alerts.alert_id.isin(case_alert_ids)].copy()
st.markdown("#### Alerts")
alert_cols=[c for c in ["alert_id","scenario_id","alert_created_at","last_trigger_at","transaction_count",
                        "alert_amount","priority","policy_flag","trigger_reason"] if c in ad.columns]
st.dataframe(ad[alert_cols].sort_values("alert_created_at"),use_container_width=True,hide_index=True)

tx=at[at.alert_id.isin(case_alert_ids)].copy()
st.markdown("#### Transaction evidence")
st.caption("Canonical ALERT_TRANSACTION evidence links. The dashboard does not infer AML outcomes.")
st.dataframe(tx.sort_values(["alert_id","linked_at"]),use_container_width=True,hide_index=True)

with st.expander("Governance / lineage"):
 st.write("Detection portfolio: **Frozen V3**")
 st.write("Operational layer: **Frozen V1**")
 st.write("Case sessionization: same primary account; new case after >30-day alert gap.")
 st.write("Priority is transparent rule-based workflow prioritization; AML outcome labels are not used.")
 st.write("Policy flag denotes governed geography-control involvement; it is not an AML determination.")
