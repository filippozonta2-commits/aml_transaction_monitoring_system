import streamlit as st

def render(d):
    q=d["queue"]
    st.header("Prioritized Investigator Queue")
    c1,c2,c3=st.columns(3)
    bands=c1.multiselect("Priority",["Critical","High","Medium","Standard"],default=["Critical","High"])
    coverage=c2.selectbox("Scenario coverage",["All","Multi-scenario only","Single-scenario only"])
    n=c3.selectbox("Rows",[100,250,500,1000],index=1)
    z=q[q.priority_band.isin(bands)].copy()
    if coverage=="Multi-scenario only": z=z[z.multi_scenario_flag.eq(1)]
    elif coverage=="Single-scenario only": z=z[z.multi_scenario_flag.eq(0)]
    cols=["priority_rank","priority_band","CASE_ID","account_id","risk_score","case_start","case_end",
          "transaction_alerts","total_alert_amount","active_scenarios","scenario_list"]
    st.dataframe(z[cols].head(n),use_container_width=True,hide_index=True)
