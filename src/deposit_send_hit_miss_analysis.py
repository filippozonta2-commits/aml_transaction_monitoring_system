"""Deposit-Send diagnostic: characterize target sequences and test account-role semantics.

No threshold tuning. Compares subsequent activity from the Cash Deposit
Sender_account versus Receiver_account for DEVELOPMENT target cases only.
HOLDOUT is never accessed.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def load(path, period):
    cols=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
          "Payment_currency","Received_currency","Sender_bank_location",
          "Receiver_bank_location","Is_laundering","Laundering_type"]
    d=pd.read_csv(path,usecols=cols)
    d["ts"]=pd.to_datetime(pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")
                           +" "+d["Time"].astype(str),errors="coerce")
    d["Period"]=period
    return d.sort_values("ts")

def future_stats(events, acct, t, hours):
    z=events[(events["Sender_account"].eq(acct)) &
             (events["ts"].gt(t)) &
             (events["ts"].le(t+pd.Timedelta(hours=hours))) &
             (~events["Payment_type"].eq("Cash Deposit"))]
    if z.empty:
        return (0,0.0,0,np.nan,"NONE")
    return (len(z),float(z["Amount"].sum()),z["Receiver_account"].nunique(),
            (z["ts"].iloc[0]-t).total_seconds()/3600,
            str(z["Payment_type"].iloc[0]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("results/deposit_send_diagnostic"))
    a=p.parse_args()

    print("Loading context + DEVELOPMENT...")
    tr=load(a.train,"train"); dev=load(a.development,"development")
    start=dev["ts"].min()
    events=pd.concat([tr[tr["ts"].ge(start-pd.Timedelta(days=7))],dev],
                     ignore_index=True).sort_values("ts")
    targets=dev[(dev["Is_laundering"].eq(1)) &
                (dev["Laundering_type"].eq("Deposit-Send"))].copy().reset_index(drop=True)
    print(f"Deposit-Send target transactions: {len(targets):,}")
    print("Testing both Sender_account and Receiver_account as the account to follow...")
    rows=[]
    for i,r in targets.iterrows():
        base={"target_id":i,"ts":r.ts,"amount":r.Amount,"payment_type":r.Payment_type,
              "sender":r.Sender_account,"receiver":r.Receiver_account}
        for role,acct in [("sender",r.Sender_account),("receiver",r.Receiver_account)]:
            for h in [24,72,168]:
                n,total,cps,first,ptype=future_stats(events,acct,r.ts,h)
                rows.append({**base,"follow_role":role,"horizon_hours":h,
                             "outgoing_count":n,"cumulative_outflow":total,
                             "outflow_ratio":total/r.Amount if r.Amount else np.nan,
                             "unique_receivers":cps,"hours_to_first":first,
                             "first_outgoing_payment_type":ptype})
    diag=pd.DataFrame(rows)
    summary=diag.groupby(["follow_role","horizon_hours"]).agg(
        Targets=("target_id","nunique"),
        Targets_with_outflow=("outgoing_count",lambda x:(x>0).sum()),
        Median_outgoing_count=("outgoing_count","median"),
        Median_outflow_ratio=("outflow_ratio","median"),
        Median_hours_to_first=("hours_to_first","median"),
        Median_unique_receivers=("unique_receivers","median")
    ).reset_index()
    summary["Coverage_with_outflow"]=summary["Targets_with_outflow"]/summary["Targets"]

    ptypes=(diag[diag["first_outgoing_payment_type"].ne("NONE")]
            .groupby(["follow_role","horizon_hours","first_outgoing_payment_type"])
            .size().reset_index(name="Targets")
            .sort_values(["follow_role","horizon_hours","Targets"],ascending=[True,True,False]))

    a.output_dir.mkdir(parents=True,exist_ok=True)
    diag.to_csv(a.output_dir/"deposit_send_target_role_features.csv",index=False)
    summary.to_csv(a.output_dir/"deposit_send_role_summary.csv",index=False)
    ptypes.to_csv(a.output_dir/"deposit_send_next_payment_types.csv",index=False)

    print("\n=== DEPOSIT-SEND ACCOUNT-ROLE DIAGNOSTIC ===")
    print(summary.to_string(index=False))
    print("\n=== MOST COMMON FIRST OUTGOING PAYMENT TYPES ===")
    print(ptypes.groupby(["follow_role","horizon_hours"],group_keys=False).head(5).to_string(index=False))
    print("\nDiagnostic only: no thresholds were selected and HOLDOUT was not accessed.")

if __name__=="__main__":
    main()
