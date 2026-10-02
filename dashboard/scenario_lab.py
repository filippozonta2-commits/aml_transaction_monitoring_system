import numpy as np
import pandas as pd
import streamlit as st
from src.scenario_config_frozen import FROZEN_SCENARIOS
from src.scenario_catalog import SCENARIO_CATALOG

FLAG_TO_CFG={
 "SCN_SMURFING":"smurfing_cash_deposit","SCN_CASH_WITHDRAWAL":"cash_withdrawal",
 "SCN_FAN_OUT":"fan_out","SCN_STRUCTURING":"structuring","SCN_FAN_IN":"fan_in",
 "SCN_DEPOSIT_SEND":"deposit_send",
}
LABELS={k:k.replace("SCN_","").replace("_"," ").title() for k in FLAG_TO_CFG}

def _ts(x): return pd.to_datetime(x["Date"].astype(str)+" "+x["Time"].astype(str),errors="coerce")
def _fixed(x,days,account,start):
    z=x.copy(); z["w"]=np.floor((z.ts-start).dt.total_seconds()/86400/days).astype("int32")
    return z,z.groupby(["w",account],sort=False)
def _sender_features(x,days,start):
    z,g=_fixed(x,days,"Sender_account",start)
    a=g.agg(tx_count=("Amount","size"),aggregate_amount=("Amount","sum"),median_amount=("Amount","median"),
            unique_receivers=("Receiver_account","nunique"),cross_border_rate=("cross","mean"),
            currency_mismatch_rate=("fx","mean")).reset_index()
    return z.merge(a,on=["w","Sender_account"],how="left")
def _receiver_features(x,days):
    z=x.copy(); z["wr"]=z.ts.dt.floor(f"{int(days)}D")
    a=z.groupby(["wr","Receiver_account"],sort=False).agg(receiver_tx_count=("Amount","size"),
      receiver_aggregate_amount=("Amount","sum"),receiver_unique_senders=("Sender_account","nunique"),
      receiver_cross_border_rate=("cross","mean"),receiver_currency_mismatch_rate=("fx","mean")).reset_index()
    return z.merge(a,on=["wr","Receiver_account"],how="left")

@st.cache_data(show_spinner="Preparing sandbox transaction features...")
def _load_raw(path="data/temporal/SAML-D_holdout.csv"):
    cols=["Date","Time","Sender_account","Receiver_account","Amount","Payment_type","Payment_currency",
          "Received_currency","Sender_bank_location","Receiver_bank_location"]
    x=pd.read_csv(path,usecols=cols); x["holdout_row_id"]=np.arange(len(x),dtype="int64"); x["ts"]=_ts(x)
    x=x.dropna(subset=["ts"]); x["cross"]=(x.Sender_bank_location!=x.Receiver_bank_location).astype("int8")
    x["fx"]=(x.Payment_currency!=x.Received_currency).astype("int8"); return x

def _control(k,v):
    if isinstance(v,int):
        hi=max(int(v*3),v+10,10); return st.number_input(k,min_value=1,max_value=hi,value=int(v),step=1)
    hi=max(float(v*3),float(v)+1,1.0); step=max(abs(float(v))/20,.01)
    return st.number_input(k,min_value=0.0,max_value=hi,value=float(v),step=step,format="%.3f")

def _simulate(flag,cfg,p):
    x=_load_raw(); start=x.ts.min()
    if flag=="SCN_SMURFING":
        z=_sender_features(x[x.Payment_type.eq("Cash Deposit")],int(p["window_days"]),start)
        m=z.tx_count.ge(p["min_count"])&z.median_amount.lt(p["median_ceiling"])&z.aggregate_amount.ge(p["aggregate_floor"])&z.cross_border_rate.le(.15)&z.currency_mismatch_rate.le(.15)
    elif flag=="SCN_CASH_WITHDRAWAL":
        z=_sender_features(x[x.Payment_type.eq("Cash Withdrawal")],int(p["window_days"]),start)
        m=z.tx_count.ge(p["min_count"])&z.aggregate_amount.ge(p["aggregate_floor"])&z.median_amount.le(p["median_ceiling"])
    elif flag=="SCN_FAN_OUT":
        z=_sender_features(x,int(p["window_days"]),start)
        m=z.unique_receivers.ge(p["min_counterparties"])&z.tx_count.between(p["min_counterparties"],p["max_transactions"])&((z.cross_border_rate.ge(p["geo_fx_threshold"]))|(z.currency_mismatch_rate.ge(p["geo_fx_threshold"])))
    elif flag=="SCN_STRUCTURING":
        z=_receiver_features(x,int(p["window_days"]))
        m=z.Amount.lt(p["transaction_amount_ceiling"])&z.receiver_unique_senders.ge(p["min_unique_senders"])&z.receiver_aggregate_amount.ge(p["aggregate_floor"])&((z.receiver_cross_border_rate.ge(p["cross_border_rate_threshold"]))|(z.receiver_currency_mismatch_rate.ge(p["currency_mismatch_rate_threshold"])))
    elif flag=="SCN_FAN_IN":
        z=_receiver_features(x,int(p["window_days"]))
        m=z.receiver_unique_senders.between(p["min_unique_senders"],p["max_unique_senders"])&z.receiver_tx_count.between(p["min_transactions"],p["max_transactions"])&((z.receiver_cross_border_rate.ge(p["cross_border_rate_threshold"]))|(z.receiver_currency_mismatch_rate.ge(p["currency_mismatch_rate_threshold"])))
    else:
        return None
    return set(z.loc[m,"holdout_row_id"].astype(int))

def render(d,mode="monitor"):
    alerts=d["alerts"]; flags=[c for c in FLAG_TO_CFG if c in alerts.columns]
    if mode=="monitor":
        st.header("Scenario Monitoring"); st.caption("RULE-BASED DETECTION · Frozen scenarios generate transaction alerts. ML is not used here.")
        name=st.selectbox("Scenario",flags,format_func=lambda x:LABELS[x]); hit=alerts[alerts[name].eq(1)].copy()
        meta=SCENARIO_CATALOG[name]
        st.markdown(f"**{meta['direction']}** · Primary entity: **{meta['entity']}**")
        st.caption(meta["pattern"])
        a,b,c=st.columns(3); a.metric("Triggered transactions",f"{len(hit):,}"); b.metric("Trigger rate",f"{len(hit)/len(alerts):.3%}"); c.metric("Engine","Frozen rules")
        st.subheader("Triggered transaction alerts")
        show=[x for x in ["holdout_row_id","ts","Sender_account","Receiver_account","Amount","Payment_type","SCENARIO_COUNT"] if x in hit.columns]
        st.dataframe(hit[show] if show else hit,width="stretch",hide_index=True)
        st.caption("Network semantics: "+meta["network"])
        return

    st.header("Simulation Lab"); st.caption("SANDBOX · Change rule thresholds and re-run detection without modifying frozen outputs or the ML model.")
    st.warning("SANDBOX — NO PRODUCTION IMPACT")
    name=st.selectbox("Scenario",flags,format_func=lambda x:LABELS[x]); key=FLAG_TO_CFG[name]; cfg=FROZEN_SCENARIOS[key]
    meta=SCENARIO_CATALOG[name]
    st.markdown(f"**{meta['direction']}** · Primary entity: **{meta['entity']}**")
    st.caption(meta["pattern"])
    frozen=set(alerts.loc[alerts[name].eq(1),"holdout_row_id"].astype(int))
    a,b=st.columns(2); a.metric("Frozen alerts",f"{len(frozen):,}"); b.metric("Frozen trigger rate",f"{len(frozen)/len(alerts):.3%}")
    st.subheader("Sandbox thresholds")
    excluded={"development_metrics","status","payment_type","candidate","geo_fx_logic","require_cross_border","horizon_hours"}
    params={}
    for k,v in cfg.items():
        if k not in excluded and isinstance(v,(int,float)): params[k]=_control(k,v)
    if name=="SCN_DEPOSIT_SEND":
        st.info("Deposit-Send uses a forward 72-hour Spark event join. Interactive re-materialization is intentionally disabled here; its frozen configuration remains visible in Governance.")
        return
    if st.button("Run simulation",type="primary"):
        with st.spinner("Re-running scenario on HOLDOUT transaction features..."): simulated=_simulate(name,cfg,params)
        retained=frozen&simulated; removed=frozen-simulated; new=simulated-frozen
        st.subheader("Frozen vs Sandbox")
        a,b,c,e=st.columns(4); a.metric("Sandbox alerts",f"{len(simulated):,}",f"{len(simulated)-len(frozen):+,}"); b.metric("Retained",f"{len(retained):,}"); c.metric("Removed",f"{len(removed):,}"); e.metric("New",f"{len(new):,}")
        delta=(len(simulated)-len(frozen))/len(frozen) if frozen else 0; st.metric("Alert-volume change",f"{delta:+.2%}")
        comp=pd.DataFrame({"holdout_row_id":list(retained)+list(removed)+list(new),"sandbox_status":["RETAINED"]*len(retained)+["REMOVED"]*len(removed)+["NEW"]*len(new)})
        detail=alerts.drop(columns=[c for c in flags if c!=name],errors="ignore").merge(comp,on="holdout_row_id",how="inner")
        st.dataframe(detail.sort_values(["sandbox_status","holdout_row_id"]),width="stretch",hide_index=True)
        st.download_button("Export sandbox comparison",detail.to_csv(index=False).encode(),"sandbox_comparison.csv","text/csv")
