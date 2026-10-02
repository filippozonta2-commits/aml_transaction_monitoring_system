"""Explore a channel-aware Deposit-Send sequence detector on DEVELOPMENT.

Links Cash Deposit activity to subsequent outgoing transactions by the same
account within short temporal windows. TRAIN is historical context only.
HOLDOUT is never accessed.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load(path, period):
    cols=["Time","Date","Sender_account","Receiver_account","Amount",
          "Payment_type","Is_laundering","Laundering_type"]
    d=pd.read_csv(path,usecols=cols)
    d["ts"]=pd.to_datetime(
        pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")
        +" "+d["Time"].astype(str),errors="coerce")
    d["Period"]=period
    return d


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("results/deposit_send_analysis"))
    a=p.parse_args()

    print("Loading TRAIN warm-up + DEVELOPMENT...")
    tr=load(a.train,"train"); dev=load(a.development,"development")
    start=dev["ts"].min()
    warm=tr[tr["ts"].ge(start-pd.Timedelta(days=14))]
    data=pd.concat([warm,dev],ignore_index=True).sort_values("ts")
    del tr,warm
    print(f"Development rows: {len(dev):,}")
    print("HOLDOUT is not accessed.")

    # In SAML-D, Cash Deposit is represented as an event sent by an account.
    # We therefore follow subsequent outgoing activity from that same account.
    deposits=data[data["Payment_type"].eq("Cash Deposit")][
        ["ts","Sender_account","Amount","Period","Is_laundering","Laundering_type"]
    ].rename(columns={"ts":"deposit_ts","Amount":"deposit_amount"})

    outgoing=data[~data["Payment_type"].eq("Cash Deposit")][
        ["ts","Sender_account","Receiver_account","Amount","Payment_type",
         "Period","Is_laundering","Laundering_type"]
    ].rename(columns={"ts":"send_ts","Amount":"send_amount"})

    # Efficient nearest-next-event linkage by account.
    deposits=deposits.sort_values(["deposit_ts","Sender_account"])
    outgoing=outgoing.sort_values(["send_ts","Sender_account"])
    pairs=pd.merge_asof(
        deposits,outgoing,left_on="deposit_ts",right_on="send_ts",
        by="Sender_account",direction="forward",
        suffixes=("_deposit","_send")
    )
    pairs["hours_to_send"]=(pairs["send_ts"]-pairs["deposit_ts"]).dt.total_seconds()/3600
    pairs["amount_ratio"]=pairs["send_amount"]/pairs["deposit_amount"].replace(0,np.nan)

    # Evaluate only development deposits; label is attached to the deposit event.
    ev=pairs[pairs["Period_deposit"].eq("development")].copy()
    ev["target"]=(
        ev["Is_laundering_deposit"].eq(1)&
        ev["Laundering_type_deposit"].eq("Deposit-Send")
    )

    print(f"Development Cash Deposits: {len(ev):,}")
    print(f"Deposit-Send target deposits: {int(ev['target'].sum()):,}")

    rows=[]
    for hours in [1,6,12,24,48,72]:
        for lo in [.25,.50,.75]:
            for hi in [1.25,1.50,2.00]:
                mask=(
                    ev["hours_to_send"].between(0,hours)&
                    ev["amount_ratio"].between(lo,hi)
                )
                trig=int(mask.sum()); hits=int((mask&ev["target"]).sum())
                target=int(ev["target"].sum())
                aml=int((mask&ev["Is_laundering_deposit"].eq(1)).sum())
                rows.append(dict(
                    Window_hours=hours,Min_amount_ratio=lo,Max_amount_ratio=hi,
                    Triggered_deposits=trig,
                    Trigger_rate=trig/len(ev) if len(ev) else np.nan,
                    AML_cases=aml,
                    Precision=aml/trig if trig else np.nan,
                    Deposit_Send_hits=hits,
                    Deposit_Send_recall=hits/target if target else np.nan,
                ))

    res=pd.DataFrame(rows)
    top=res[res["Trigger_rate"].le(.10)].sort_values(
        ["Deposit_Send_recall","Precision","Trigger_rate"],
        ascending=[False,False,True]).head(15)

    a.output_dir.mkdir(parents=True,exist_ok=True)
    res.to_csv(a.output_dir/"deposit_send_grid.csv",index=False)
    top.to_csv(a.output_dir/"deposit_send_shortlist.csv",index=False)
    ev[ev["target"]].to_csv(a.output_dir/"deposit_send_target_sequences.csv",index=False)

    print("\n=== DEPOSIT-SEND SCENARIO SHORTLIST ===")
    print(top.to_string(index=False))
    print("\nThis V1 links each Cash Deposit to the next outgoing transaction from the same account.")
    print("All metrics are DEVELOPMENT-only. HOLDOUT was not accessed.")


if __name__=="__main__":
    main()
