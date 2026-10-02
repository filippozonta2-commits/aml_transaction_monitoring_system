"""Materialize frozen non-Deposit-Send scenario flags on DEVELOPMENT.

Uses the exact fixed-window semantics used during each DEVELOPMENT freeze/tuning
step. TRAIN is warm-up context only. HOLDOUT is never read.

Deposit-Send remains separate because its receiver-flow range join is executed
by the validated Spark implementation; it will be merged into the unified alert
table in the next step.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scenario_config_frozen import FROZEN_SCENARIOS

COLS=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
      "Payment_currency","Received_currency","Sender_bank_location",
      "Receiver_bank_location","Is_laundering","Laundering_type"]

def load(path,period):
    d=pd.read_csv(path,usecols=COLS)
    d["ts"]=pd.to_datetime(pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")
                           +" "+d["Time"].astype(str),errors="coerce")
    d=d.dropna(subset=["ts"])
    d["Period"]=period
    d["cross"]=(d["Sender_bank_location"]!=d["Receiver_bank_location"]).astype("int8")
    d["fx"]=(d["Payment_currency"]!=d["Received_currency"]).astype("int8")
    return d

def sender_windows(data,start,days):
    x=data.copy()
    x["window_id"]=np.floor((x["ts"]-start).dt.total_seconds()/86400/days).astype("int32")
    g=x.groupby(["window_id","Sender_account"],observed=True).agg(
        tx_count=("Amount","size"),aggregate_amount=("Amount","sum"),
        median_amount=("Amount","median"),unique_receivers=("Receiver_account","nunique"),
        cross_border_rate=("cross","mean"),currency_mismatch_rate=("fx","mean")).reset_index()
    return x.merge(g,on=["window_id","Sender_account"],how="left")

def receiver_windows(dev,days=10):
    x=dev.copy(); x["window_id_r"]=x["ts"].dt.floor(f"{days}D")
    g=x.groupby(["window_id_r","Receiver_account"],observed=True).agg(
        receiver_tx_count=("Amount","size"),receiver_aggregate_amount=("Amount","sum"),
        receiver_unique_senders=("Sender_account","nunique"),
        receiver_cross_border_rate=("cross","mean"),
        receiver_currency_mismatch_rate=("fx","mean")).reset_index()
    return x.merge(g,on=["window_id_r","Receiver_account"],how="left")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("results/unified_dashboard"))
    a=p.parse_args()
    print("Loading TRAIN warm-up + DEVELOPMENT...")
    tr=load(a.train,"train"); dev=load(a.development,"development")
    start=dev["ts"].min()
    warm=tr[tr["ts"].ge(start-pd.Timedelta(days=60))]
    data=pd.concat([warm,dev],ignore_index=True)
    print(f"Development rows: {len(dev):,}")
    print("HOLDOUT is not accessed.")

    # Stable key based on original development row order.
    dev=dev.copy(); dev["development_row_id"]=np.arange(len(dev),dtype="int64")
    # Rebuild combined frame so row ids survive only on development rows.
    warm=warm.copy(); warm["development_row_id"]=-1
    data=pd.concat([warm,dev],ignore_index=True)

    flags=pd.DataFrame({"development_row_id":dev["development_row_id"]})

    # Cash Withdrawal: 7D sender windows, channel-specific.
    cw=data[data["Payment_type"].eq("Cash Withdrawal")].copy()
    cw=sender_windows(cw,start,7)
    m=(cw["tx_count"].ge(5)&cw["aggregate_amount"].ge(300)&cw["median_amount"].le(300))
    ids=set(cw.loc[m & cw["Period"].eq("development"),"development_row_id"].astype(int))
    flags["SCN_CASH_WITHDRAWAL"]=flags["development_row_id"].isin(ids).astype("int8")

    # Smurfing Cash Deposit: exact frozen 45D channel-aware rule.
    sm=data[data["Payment_type"].eq("Cash Deposit")].copy()
    sm=sender_windows(sm,start,45)
    m=(
        sm["tx_count"].ge(3)
        & sm["median_amount"].lt(4000)
        & sm["aggregate_amount"].ge(10000)
        & sm["cross_border_rate"].le(.15)
        & sm["currency_mismatch_rate"].le(.15)
    )
    ids=set(sm.loc[m & sm["Period"].eq("development"),"development_row_id"].astype(int))
    flags["SCN_SMURFING"]=flags["development_row_id"].isin(ids).astype("int8")

    # Fan-Out: frozen 21D sender-window rule from V2.2.
    fo=sender_windows(data,start,21)
    m=(fo["unique_receivers"].ge(3)&fo["tx_count"].between(3,12)&
       ((fo["cross_border_rate"].ge(.20))|(fo["currency_mismatch_rate"].ge(.20))))
    ids=set(fo.loc[m & fo["Period"].eq("development"),"development_row_id"].astype(int))
    flags["SCN_FAN_OUT"]=flags["development_row_id"].isin(ids).astype("int8")

    # Structuring + Fan-In: exact DEVELOPMENT 10D fixed buckets used at freeze.
    r=receiver_windows(dev,10)
    flags["SCN_STRUCTURING"]=(
        r["Amount"].lt(10000)&r["receiver_unique_senders"].ge(5)&
        r["receiver_aggregate_amount"].ge(20000)&
        ((r["receiver_cross_border_rate"].ge(.20))|
         (r["receiver_currency_mismatch_rate"].ge(.25)))).astype("int8").to_numpy()
    flags["SCN_FAN_IN"]=(
        r["receiver_unique_senders"].between(5,15)&r["receiver_tx_count"].between(5,20)&
        ((r["receiver_cross_border_rate"].ge(.10))|
         (r["receiver_currency_mismatch_rate"].ge(.20)))).astype("int8").to_numpy()

    meta=dev[["development_row_id","ts","Sender_account","Receiver_account","Amount",
              "Payment_type","Is_laundering","Laundering_type"]].reset_index(drop=True)
    out=meta.merge(flags,on="development_row_id",how="left")
    flagcols=[c for c in flags.columns if c.startswith("SCN_")]
    out["SCENARIO_COUNT"]=out[flagcols].sum(axis=1)
    out["ANY_SCENARIO_ALERT"]=(out["SCENARIO_COUNT"]>0).astype("int8")

    rows=[]
    target_map={"SCN_SMURFING":"Smurfing","SCN_CASH_WITHDRAWAL":"Cash_Withdrawal",
                "SCN_FAN_OUT":"Fan_Out","SCN_STRUCTURING":"Structuring","SCN_FAN_IN":"Fan_In"}
    for f,t in target_map.items():
        hit=out[f].eq(1); aml=out["Is_laundering"].eq(1); target=aml&out["Laundering_type"].eq(t)
        rows.append({"Scenario":f,"Triggered":int(hit.sum()),"Trigger_rate":hit.mean(),
                     "AML_cases":int((hit&aml).sum()),"Precision":(hit&aml).sum()/hit.sum() if hit.sum() else np.nan,
                     "Target":t,"Target_hits":int((hit&target).sum()),"Target_total":int(target.sum()),
                     "Target_recall":(hit&target).sum()/target.sum() if target.sum() else np.nan})
    summary=pd.DataFrame(rows)

    a.output_dir.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output_dir/"materialized_non_deposit_send.csv",index=False)
    summary.to_csv(a.output_dir/"materialized_non_deposit_send_summary.csv",index=False)
    print("\n=== MATERIALIZED FROZEN SCENARIOS ===")
    print(summary.to_string(index=False))
    print(f"\nAny-scenario alerts: {int(out['ANY_SCENARIO_ALERT'].sum()):,}")
    print(f"Saved: {a.output_dir}")
    print("Development-only. HOLDOUT was not accessed.")

if __name__=="__main__": main()
