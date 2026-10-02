import pandas as pd
import streamlit as st
from src.scenario_config_frozen import FROZEN_SCENARIOS

SKIP={"development_metrics","status","payment_type","candidate","geo_fx_logic","require_cross_border"}

def render(d):
    st.header("Scenario Explorer")
    st.caption("Explore the alerts produced by each frozen transaction-monitoring scenario.")
    alerts=d["alerts"]; sc=d["scenarios"]
    flags=[c for c in alerts.columns if c.startswith("SCN_")]
    name=st.selectbox("Scenario",flags)
    hit=alerts[alerts[name].eq(1)].copy()
    frozen=int(hit.shape[0])
    meta=sc[sc["scenario"].eq(name)]
    c1,c2,c3=st.columns(3)
    c1.metric("Triggered transactions",f"{frozen:,}")
    c2.metric("Trigger rate",f"{frozen/len(alerts):.3%}")
    c3.metric("Share of all transactions",f"{frozen/len(alerts):.3%}")
    st.subheader("Triggered alerts")
    show=[x for x in ["holdout_row_id","Date","Time","Sender_account","Receiver_account","Amount",
                       "Payment_type","Sender_bank_location","Receiver_bank_location","SCENARIO_COUNT"] if x in hit.columns]
    if show:
        st.dataframe(hit[show],width="stretch",hide_index=True)
    else:
        st.dataframe(hit,width="stretch",hide_index=True)
    st.download_button("Export selected alerts",hit.to_csv(index=False).encode(),"scenario_alerts.csv","text/csv")

    st.divider()
    st.subheader("Threshold what-if")
    st.warning("Sandbox only. Threshold controls do not alter the frozen model, frozen scenarios, or official HOLDOUT evaluation.")
    cfg=FROZEN_SCENARIOS.get(name,{})
    editable=[(k,v) for k,v in cfg.items() if k not in SKIP and isinstance(v,(int,float))]
    if not editable:
        st.info("This scenario has no numeric frozen thresholds exposed in the current configuration.")
    else:
        for key,val in editable:
            val=float(val); hi=max(val*3,1.0); step=max(val/20,.01)
            st.slider(key,0.0,hi,val,step,key=f"lab_{name}_{key}")
        st.caption("Controls are visible for scenario design review. Recalculation will be enabled only from non-label transaction features, so AML truth never enters the interactive decision path.")
