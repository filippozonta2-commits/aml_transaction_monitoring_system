import pandas as pd
import streamlit as st

def render(d):
    st.header("Detection Model Comparison")
    st.caption("Reported frozen Rule-Based V3 vs XGBoost V1 · identical documented HOLDOUT population · independent reproduction pending")
    m=d["metrics"]; rp=m[m.scenario.eq("PORTFOLIO_V3")].iloc[0]
    ml=d.get("ml_detection"); ov=d.get("overlap")
    if ml is None or ov is None:
        st.warning("Frozen ML comparison outputs are missing. Run evaluate_frozen_ml_v1_holdout.py and compare_frozen_rule_v3_ml_v1.py.")
        return
    n=len(ml); y=int(ml["Is_laundering"].sum()); ma=int(ml["ML_ALERT_V1"].sum())
    mh=int(ml.loc[ml.ML_ALERT_V1.eq(1),"Is_laundering"].sum())
    if not y or not ma or not int(rp.alerts):
        st.error("Cannot compute comparison metrics: empty positive class or alert population.")
        return
    mr=mh/y; mp=mh/ma
    if "aml_hits" in rp and int(rp.aml_hits) > y:
        st.error("Rule and ML results appear inconsistent: Rule AML hits exceed ML HOLDOUT positives. Check dataset/version alignment.")
        return
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Rule V3 recall",f"{rp.recall:.2%}")
    c2.metric("ML V1 recall",f"{mr:.2%}",delta=f"{(mr-rp.recall)*100:+.2f} pp")
    c3.metric("Rule V3 alerts",f"{int(rp.alerts):,}")
    c4.metric("ML V1 alerts",f"{ma:,}",delta=f"{(ma/int(rp.alerts)-1):.1%}")
    st.subheader("Frozen HOLDOUT performance")
    perf=pd.DataFrame([
      {"Detector":"Rule-Based V3","Alerts":int(rp.alerts),"Alert rate":float(rp.alert_rate),"AML hits":int(rp.aml_hits),"Precision":float(rp.precision),"Recall":float(rp.recall)},
      {"Detector":"ML Detection V1","Alerts":ma,"Alert rate":ma/n,"AML hits":mh,"Precision":mp,"Recall":mr},
    ])
    st.dataframe(perf,width="stretch",hide_index=True,column_config={
      "Alert rate":st.column_config.NumberColumn(format="%.2%%"),
      "Precision":st.column_config.NumberColumn(format="%.2%%"),
      "Recall":st.column_config.NumberColumn(format="%.2%%")})
    left,right=st.columns(2)
    with left:
        st.subheader("AML detection overlap")
        x=ov[["group","aml_positives"]].copy()
        st.bar_chart(x.set_index("group")["aml_positives"])
    with right:
        st.subheader("Transaction workload overlap")
        x=ov[["group","transactions"]].copy()
        st.bar_chart(x.set_index("group")["transactions"])
    st.info("Overlap is a post-HOLDOUT diagnostic only, not a frozen combined detector. The charts above are calculated from the saved comparison artifact. Original model/feature/threshold provenance is not independently reproducible from the public checkout; see docs/independent_validation_audit_2026-10-29.md.")
    st.subheader("What the comparison means")
    st.markdown(f"""**ML V1** reduces alert volume from **{int(rp.alerts):,}** to **{ma:,}** while increasing captured AML from **{int(rp.aml_hits):,}** to **{mh:,}**.  
**Rule-Based V3** remains the transparent policy/scenario benchmark and retains limited complementary coverage.  
The two frozen systems must remain unchanged; HOLDOUT results are validation evidence, not tuning input.""")
