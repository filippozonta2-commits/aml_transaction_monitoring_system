import streamlit as st
from src.scenario_config_frozen import FROZEN_SCENARIOS

SKIP={"development_metrics","status","payment_type","candidate","geo_fx_logic","require_cross_border"}

def render(_d):
    st.header("Scenario Lab")
    st.warning("What-if workspace only. Changes here never alter the frozen production configuration or official HOLDOUT results.")
    name=st.selectbox("Scenario",list(FROZEN_SCENARIOS))
    cfg=FROZEN_SCENARIOS[name]
    st.write("Adjust a copy of the frozen thresholds:")
    for key,val in cfg.items():
        if key in SKIP or not isinstance(val,(int,float)): continue
        val=float(val); hi=max(val*3,1.0); step=max(val/20,0.01)
        st.slider(key,0.0,hi,val,step,key=f"lab_{name}_{key}")
    st.info("Next step: connect these controls to a simulation engine that recomputes alert volume, case workload and maps without exposing AML truth.")
