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
    alerts=d["alerts"]
    if mode=="monitor":
        st.header("Scenario Monitoring")
        st.caption("FROZEN V3 · Operational alert monitoring. Detection thresholds are not recalculated here.")
        scenarios=sorted(alerts["scenario_id"].dropna().unique())
        name=st.selectbox("Scenario",scenarios,format_func=lambda x:x.replace("SCN_","").replace("_"," ").title())
        hit=alerts[alerts["scenario_id"].eq(name)].copy()
        total_tx=int(hit["transaction_count"].sum())
        a,b,c=st.columns(3)
        a.metric("Aggregated alerts",f"{len(hit):,}")
        b.metric("Triggering transactions",f"{total_tx:,}")
        c.metric("Unique accounts",f"{hit['primary_account_id'].nunique():,}")
        st.subheader("Operational alert population")
        show=[x for x in ["alert_id","primary_account_id","alert_created_at","last_trigger_at","transaction_count","alert_amount","policy_flag","trigger_reason"] if x in hit.columns]
        st.dataframe(hit[show].sort_values("alert_created_at"),use_container_width=True,hide_index=True)
        st.caption("These are Frozen V3 triggers after Operational Layer V1 alert aggregation.")
        return

    st.header("Simulation Lab")
    st.caption("Legacy threshold sandbox is intentionally separated from Frozen V3 Operational Layer V1.")
    st.warning("The previous interactive simulator was built against the pre-V3 transaction-alert schema. It is disabled here to prevent mixing legacy outputs with the frozen V3 operational dashboard.")
    st.info("Scenario redesign should be performed in a new governed version using DEVELOPMENT data, then validated once on a fresh HOLDOUT.")

