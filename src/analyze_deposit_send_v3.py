"""Deposit-Send V3: receiver-based flow-of-funds scenario candidates.

Development-only calibration. Follows the RECEIVER of every Deposit-Send-like
starting transaction and measures subsequent outgoing activity. This avoids
assuming that the typology name implies Payment_type == Cash Deposit.
HOLDOUT is never accessed.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def load(path, period):
    cols=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
          "Is_laundering","Laundering_type"]
    d=pd.read_csv(path,usecols=cols)
    d["ts"]=pd.to_datetime(
        pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")
        +" "+d["Time"].astype(str),errors="coerce")
    d["Period"]=period
    return d.sort_values("ts")

def build_receiver_features(base, events, hours):
    groups={k:g.sort_values("ts") for k,g in events.groupby("Sender_account",sort=False)}
    rows=[]
    for r in base.itertuples():
        g=groups.get(r.Receiver_account)
        if g is None:
            rows.append([r.tx_id,0,0.0,0,np.nan,0,0.0])
            continue
        z=g[(g["ts"]>r.ts)&(g["ts"]<=r.ts+pd.Timedelta(hours=hours))]
        if z.empty:
            rows.append([r.tx_id,0,0.0,0,np.nan,0,0.0])
            continue
        cross=z["Payment_type"].eq("Cross-border")
        rows.append([
            r.tx_id,len(z),float(z["Amount"].sum()),z["Receiver_account"].nunique(),
            (z["ts"].iloc[0]-r.ts).total_seconds()/3600,
            int(cross.sum()),float(z.loc[cross,"Amount"].sum())
        ])
    return pd.DataFrame(rows,columns=["tx_id","outgoing_count","cumulative_outflow",
        "unique_receivers","hours_to_first","cross_border_count","cross_border_outflow"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("results/deposit_send_v3"))
    a=p.parse_args()

    print("Loading TRAIN warm-up + DEVELOPMENT...")
    tr=load(a.train,"train"); dev=load(a.development,"development")
    start=dev["ts"].min()
    events=pd.concat([tr[tr["ts"].ge(start-pd.Timedelta(days=7))],dev],
                     ignore_index=True).sort_values("ts")
    print(f"Development rows: {len(dev):,}")
    print("HOLDOUT is not accessed.")

    # Candidate starting transactions: all development transactions.
    # Labels are used only for development evaluation.
    base=dev.reset_index(drop=True).copy()
    base["tx_id"]=np.arange(len(base))
    base["target"]=(base["Is_laundering"].eq(1)&
                    base["Laundering_type"].eq("Deposit-Send"))
    target_n=int(base["target"].sum())
    target_accts=base.loc[base["target"],"Receiver_account"].nunique()
    print(f"Deposit-Send targets: {target_n:,} | target receiver accounts: {target_accts:,}")

    configs=[
        # Motivated by diagnostic: receiver outflow commonly appears by 72h,
        # with cumulative flow approaching the starting amount by 7d.
        (72,1,.50,1.50,False),
        (72,1,.50,2.00,False),
        (72,1,.25,2.00,False),
        (72,1,.50,3.00,False),
        (168,1,.50,1.50,False),
        (168,1,.50,2.00,False),
        (168,1,.25,2.00,False),
        (168,1,.50,3.00,False),
        # Cross-border variants, based on diagnostic first-outgoing distribution.
        (72,1,.25,3.00,True),
        (72,1,.50,3.00,True),
        (168,1,.25,3.00,True),
        (168,1,.50,3.00,True),
    ]

    frames={}; rows=[]
    for hours in [72,168]:
        print(f"Building receiver flow features: {hours}h...")
        f=build_receiver_features(base[["tx_id","ts","Receiver_account"]],events,hours)
        x=base[["tx_id","Amount","target","Is_laundering","Receiver_account"]].merge(f,on="tx_id")
        x["outflow_ratio"]=x["cumulative_outflow"]/x["Amount"].replace(0,np.nan)
        frames[hours]=x

    for hours,min_count,lo,hi,require_cb in configs:
        x=frames[hours]
        m=(x["outgoing_count"].ge(min_count)&x["outflow_ratio"].between(lo,hi))
        if require_cb:
            m &= x["cross_border_count"].ge(1)
        trig=int(m.sum()); hits=int((m&x["target"]).sum())
        aml=int((m&x["Is_laundering"].eq(1)).sum())
        hit_accts=x.loc[m&x["target"],"Receiver_account"].nunique()
        rows.append(dict(
            Horizon_hours=hours,Min_outgoing_count=min_count,
            Min_outflow_ratio=lo,Max_outflow_ratio=hi,
            Require_cross_border=require_cb,
            Triggered_transactions=trig,
            Trigger_rate=trig/len(x),
            AML_cases=aml,
            Precision=aml/trig if trig else np.nan,
            Deposit_Send_hits=hits,
            Deposit_Send_recall=hits/target_n if target_n else np.nan,
            Deposit_Send_hit_accounts=hit_accts,
            Deposit_Send_account_recall=hit_accts/target_accts if target_accts else np.nan
        ))

    res=pd.DataFrame(rows).sort_values(
        ["Deposit_Send_recall","Precision","Trigger_rate"],
        ascending=[False,False,True])

    a.output_dir.mkdir(parents=True,exist_ok=True)
    res.to_csv(a.output_dir/"deposit_send_v3_candidates.csv",index=False)
    for h,x in frames.items():
        x[x["target"]].to_csv(a.output_dir/f"deposit_send_target_features_{h}h.csv",index=False)

    print("\n=== DEPOSIT-SEND V3 RECEIVER-BASED CANDIDATES ===")
    print(res.to_string(index=False))
    print("\nThese are a small set of diagnostic-motivated DEVELOPMENT candidates.")
    print("No broad grid search was performed. HOLDOUT was not accessed.")

if __name__=="__main__":
    main()
