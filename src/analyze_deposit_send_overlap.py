"""Deposit-Send ensemble overlap: Cash-Deposit V2 vs receiver-flow V3.

Development-only diagnostic. HOLDOUT is never accessed.
V2: Cash Deposit, 90D, min_count=3, median<=4000, aggregate>=10000.
V3: receiver outflow within 72h, ratio 0.5-3.0, >=1 Cross-border outflow.
"""
import argparse
import pandas as pd
import numpy as np

COLS=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
      "Is_laundering","Laundering_type"]

def read(path):
    d=pd.read_csv(path,usecols=COLS)
    d["ts"]=pd.to_datetime(pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")
                           +" "+d["Time"].astype(str),errors="coerce")
    d["Amount"]=pd.to_numeric(d["Amount"],errors="coerce")
    return d.sort_values("ts").reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",default="data/temporal/SAML-D_train.csv")
    ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
    ap.add_argument("--output",default="results/deposit_send_ensemble/overlap_summary.csv")
    a=ap.parse_args()

    print("Loading TRAIN + DEVELOPMENT...")
    tr=read(a.train); dev=read(a.development)
    start=dev.ts.min()
    events=pd.concat([tr[tr.ts>=start-pd.Timedelta(days=90)],dev],
                     ignore_index=True).sort_values("ts")

    targets=dev[(dev.Is_laundering==1)&(dev.Laundering_type=="Deposit-Send")].copy()
    targets["target_id"]=np.arange(len(targets))
    print(f"Deposit-Send targets: {len(targets):,}")

    # V2 Cash-Deposit-aware: aggregate deposits made by the same sender in prior 90D.
    cash=events[events.Payment_type=="Cash Deposit"].copy()
    cash_groups={k:g.sort_values("ts") for k,g in cash.groupby("Sender_account",sort=False)}
    v2=[]
    for r in targets.itertuples():
        g=cash_groups.get(r.Sender_account)
        if g is None: v2.append(False); continue
        z=g[(g.ts>r.ts-pd.Timedelta(days=90))&(g.ts<=r.ts)]
        hit=(len(z)>=3 and z.Amount.median()<=4000 and z.Amount.sum()>=10000)
        v2.append(bool(hit))
    targets["v2_hit"]=v2

    # V3 validated receiver role: outgoing flow from deposit receiver in next 72h.
    out_groups={k:g.sort_values("ts") for k,g in events.groupby("Sender_account",sort=False)}
    v3=[]
    for r in targets.itertuples():
        g=out_groups.get(r.Receiver_account)
        if g is None: v3.append(False); continue
        z=g[(g.ts>r.ts)&(g.ts<=r.ts+pd.Timedelta(hours=72))]
        ratio=z.Amount.sum()/r.Amount if r.Amount else np.nan
        hit=(len(z)>=1 and .5<=ratio<=3.0 and (z.Payment_type=="Cross-border").any())
        v3.append(bool(hit))
    targets["v3_hit"]=v3
    targets["ensemble_hit"]=targets.v2_hit|targets.v3_hit

    n=len(targets)
    both=int((targets.v2_hit&targets.v3_hit).sum())
    v2only=int((targets.v2_hit&~targets.v3_hit).sum())
    v3only=int((~targets.v2_hit&targets.v3_hit).sum())
    neither=int((~targets.v2_hit&~targets.v3_hit).sum())
    rows=[
        ("V2 Cash-Deposit-aware",int(targets.v2_hit.sum()),targets.v2_hit.mean()),
        ("V3 receiver-flow cross-border",int(targets.v3_hit.sum()),targets.v3_hit.mean()),
        ("Ensemble union",int(targets.ensemble_hit.sum()),targets.ensemble_hit.mean()),
        ("Overlap both",both,both/n),
        ("V2 incremental only",v2only,v2only/n),
        ("V3 incremental only",v3only,v3only/n),
        ("Neither",neither,neither/n),
    ]
    out=pd.DataFrame(rows,columns=["Metric","Deposit_Send_hits","Share_of_targets"])
    from pathlib import Path
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    targets[["target_id","ts","Sender_account","Receiver_account","Amount",
             "v2_hit","v3_hit","ensemble_hit"]].to_csv(
        str(Path(a.output).parent/"target_overlap.csv"),index=False)

    print("\n=== DEPOSIT-SEND V2 + V3 OVERLAP ===")
    print(out.to_string(index=False))
    print(f"\nV3 incremental hits beyond V2: {v3only:,}")
    print(f"Ensemble recall: {targets.ensemble_hit.mean():.2%}")
    print(f"Saved: {a.output}")
    print("Development-only. HOLDOUT was not accessed.")

if __name__=="__main__":
    main()
