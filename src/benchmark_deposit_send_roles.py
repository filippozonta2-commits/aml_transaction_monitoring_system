"""Deposit-Send role benchmark: sender-flow vs receiver-flow on identical DEVELOPMENT targets.

Compares the two account-role hypotheses on the same Deposit-Send transaction
population. This intentionally does NOT call the sender-role detector "V2":
the historical V2 was anchored only on Cash Deposit rows and therefore used a
different denominator. HOLDOUT is never accessed.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

COLS=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
      "Is_laundering","Laundering_type"]

def read(path):
    d=pd.read_csv(path,usecols=COLS)
    d["ts"]=pd.to_datetime(
        pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")
        +" "+d["Time"].astype(str),errors="coerce")
    d["Amount"]=pd.to_numeric(d["Amount"],errors="coerce")
    return d.sort_values("ts").reset_index(drop=True)

def role_hits(targets, events, role, hours, lo, hi, require_cb):
    groups={k:g.sort_values("ts") for k,g in events.groupby("Sender_account",sort=False)}
    hits=[]; counts=[]; ratios=[]; cb_flags=[]
    account_col="Sender_account" if role=="sender" else "Receiver_account"
    for r in targets.itertuples():
        acct=getattr(r,account_col)
        g=groups.get(acct)
        if g is None:
            hits.append(False); counts.append(0); ratios.append(0.0); cb_flags.append(False)
            continue
        z=g[(g.ts>r.ts)&(g.ts<=r.ts+pd.Timedelta(hours=hours))]
        count=len(z)
        ratio=(z.Amount.sum()/r.Amount) if pd.notna(r.Amount) and r.Amount!=0 else np.nan
        cb=bool((z.Payment_type=="Cross-border").any())
        hit=count>=1 and pd.notna(ratio) and lo<=ratio<=hi and (cb if require_cb else True)
        hits.append(bool(hit)); counts.append(count); ratios.append(ratio); cb_flags.append(cb)
    return hits,counts,ratios,cb_flags

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",default="data/temporal/SAML-D_train.csv")
    ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
    ap.add_argument("--output-dir",default="results/deposit_send_role_benchmark")
    ap.add_argument("--hours",type=int,default=72)
    ap.add_argument("--min-ratio",type=float,default=.5)
    ap.add_argument("--max-ratio",type=float,default=3.0)
    ap.add_argument("--require-cross-border",action=argparse.BooleanOptionalAction,default=True)
    a=ap.parse_args()

    print("Loading TRAIN context + DEVELOPMENT...")
    tr=read(a.train); dev=read(a.development)
    start=dev.ts.min()
    context=tr[tr.ts>=start-pd.Timedelta(hours=a.hours)]
    events=pd.concat([context,dev],ignore_index=True).sort_values("ts")
    print("HOLDOUT is not accessed.")

    targets=dev[(dev.Is_laundering==1)&(dev.Laundering_type=="Deposit-Send")].copy().reset_index(drop=True)
    targets["target_id"]=np.arange(len(targets))
    print(f"Deposit-Send targets on common denominator: {len(targets):,}")

    for role in ("sender","receiver"):
        h,c,r,cb=role_hits(targets,events,role,a.hours,a.min_ratio,a.max_ratio,a.require_cross_border)
        targets[f"{role}_hit"]=h
        targets[f"{role}_outgoing_count"]=c
        targets[f"{role}_outflow_ratio"]=r
        targets[f"{role}_cross_border"]=cb

    targets["union_hit"]=targets.sender_hit|targets.receiver_hit
    both=targets.sender_hit&targets.receiver_hit
    sender_only=targets.sender_hit&~targets.receiver_hit
    receiver_only=~targets.sender_hit&targets.receiver_hit
    neither=~targets.sender_hit&~targets.receiver_hit
    n=len(targets)

    rows=[
        ("Sender-role flow",int(targets.sender_hit.sum()),float(targets.sender_hit.mean())),
        ("Receiver-role flow",int(targets.receiver_hit.sum()),float(targets.receiver_hit.mean())),
        ("Union",int(targets.union_hit.sum()),float(targets.union_hit.mean())),
        ("Overlap both",int(both.sum()),float(both.mean())),
        ("Sender only",int(sender_only.sum()),float(sender_only.mean())),
        ("Receiver only",int(receiver_only.sum()),float(receiver_only.mean())),
        ("Neither",int(neither.sum()),float(neither.mean())),
    ]
    summary=pd.DataFrame(rows,columns=["Metric","Deposit_Send_hits","Share_of_targets"])

    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/"role_benchmark_summary.csv",index=False)
    targets.to_csv(out/"role_benchmark_targets.csv",index=False)

    print("\n=== DEPOSIT-SEND COMMON-DENOMINATOR ROLE BENCHMARK ===")
    print(summary.to_string(index=False))
    print(f"\nCommon target denominator: {n:,}")
    print(f"Window: {a.hours}h | outflow ratio: {a.min_ratio}-{a.max_ratio} | require cross-border: {a.require_cross_border}")
    print(f"Saved: {out}")
    print("Development-only. HOLDOUT was not accessed.")

if __name__=="__main__":
    main()
