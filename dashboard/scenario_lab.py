import streamlit as st
from src.scenario_config_frozen import FROZEN_SCENARIOS

SKIP={"development_metrics","status","payment_type","candidate","geo_fx_logic","require_cross_border"}

def render(d,mode="monitor"):
    alerts=d["alerts"]; flags=[c for c in alerts.columns if c.startswith("SCN_")]
    if mode=="monitor":
        st.header("Scenario Monitoring")
        st.caption("RULE-BASED DETECTION · Frozen scenarios generate transaction alerts. ML is not used here.")
        name=st.selectbox("Scenario",flags); hit=alerts[alerts[name].eq(1)].copy()
        a,b,c=st.columns(3); a.metric("Triggered transactions",f"{len(hit):,}"); b.metric("Trigger rate",f"{len(hit)/len(alerts):.3%}"); c.metric("Engine","Frozen rules")
        st.subheader("Triggered transaction alerts")
        show=[x for x in ["holdout_row_id","Date","Time","Sender_account","Receiver_account","Amount","Payment_type","Sender_bank_location","Receiver_bank_location","SCENARIO_COUNT"] if x in hit.columns]
        st.dataframe(hit[show] if show else hit,width="stretch",hide_index=True)
        st.download_button("Export selected alerts",hit.to_csv(index=False).encode(),"scenario_alerts.csv","text/csv")
        return

    st.header("Simulation Lab")
    st.caption("SANDBOX · Explore alternative thresholds without changing the frozen scenarios, ML model, or official HOLDOUT results.")
    st.warning("SANDBOX — NO PRODUCTION IMPACT")
    name=st.selectbox("Scenario to simulate",flags)
    hit=alerts[alerts[name].eq(1)]
    a,b=st.columns(2); a.metric("Frozen alerts",f"{len(hit):,}"); b.metric("Frozen trigger rate",f"{len(hit)/len(alerts):.3%}")
    cfg=FROZEN_SCENARIOS.get(name,{})
    editable=[(k,v) for k,v in cfg.items() if k not in SKIP and isinstance(v,(int,float))]
    st.subheader("Sandbox configuration")
    if not editable: st.info("No numeric frozen thresholds are exposed for this scenario.")
    else:
        for key,val in editable:
            val=float(val); st.slider(key,0.0,max(val*3,1.0),val,max(val/20,.01),key=f"sim_{name}_{key}")
        st.info("Threshold editing is isolated. Dynamic re-materialization of simulated alerts is the next sandbox component; these controls do not alter frozen outputs.")
