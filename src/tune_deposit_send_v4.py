"""Deposit-Send V4 receiver-flow tuning on DEVELOPMENT only.

Uses the validated receiver-account interpretation and searches a compact grid
over horizon, cumulative outflow ratio, minimum outgoing count/receivers, and
cross-border requirement. TRAIN is used only as warm-up context. HOLDOUT is
never accessed.
"""
from pathlib import Path
import argparse, itertools
import numpy as np
import pandas as pd

COLS=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
      "Is_laundering","Laundering_type"]

def read(path):
    d=pd.read_csv(path,usecols=COLS)
    d["ts"]=pd.to_datetime(pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")+" "+d["Time"].astype(str),errors="coerce")
    d["Amount"]=pd.to_numeric(d["Amount"],errors="coerce")
    return d.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

def build_features(targets, events, hours):
    groups={k:g.sort_values("ts") for k,g in events.groupby("Sender_account",sort=False)}
    rows=[]
    for r in targets.itertuples():
        g=groups.get(r.Receiver_account)
        if g is None:
            rows.append((r.target_id,0,0,0.,False)); continue
        z=g[(g.ts>r.ts)&(g.ts<=r.ts+pd.Timedelta(hours=hours))]
        ratio=z.Amount.sum()/r.Amount if pd.notna(r.Amount) and r.Amount>0 else np.nan
        rows.append((r.target_id,len(z),z.Receiver_account.nunique(),ratio,bool((z.Payment_type=="Cross-border").any())))
    return pd.DataFrame(rows,columns=["target_id","out_count","receivers","ratio","has_cb"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",default="data/temporal/SAML-D_train.csv")
    ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
    ap.add_argument("--output-dir",default="results/deposit_send_v4")
    a=ap.parse_args()
    print("Loading TRAIN context + DEVELOPMENT...")
    tr=read(a.train); dev=read(a.development)
    print(f"Development rows: {len(dev):,}")
    print("HOLDOUT is not accessed.")
    targets=dev[(dev.Is_laundering==1)&(dev.Laundering_type=="Deposit-Send")].copy().reset_index(drop=True)
    targets["target_id"]=np.arange(len(targets))
    print(f"Deposit-Send targets: {len(targets):,}")

    horizons=[24,48,72,120,168]
    feats={}
    for h in horizons:
        print(f"Building receiver-flow features: {h}h...")
        start=dev.ts.min()
        context=tr[tr.ts>=start-pd.Timedelta(hours=h)]
        events=pd.concat([context,dev],ignore_index=True).sort_values("ts")
        feats[h]=build_features(targets,events,h)

    rows=[]
    for h,lo,hi,minc,minr,cb in itertools.product(
        horizons,[.1,.25,.5,.75],[1.25,1.5,2.,3.,5.],[1,2,3],[1,2,3],[False,True]):
        if lo>hi: continue
        f=feats[h]
        hit=(f.out_count>=minc)&(f.receivers>=minr)&f.ratio.between(lo,hi)
        if cb: hit &= f.has_cb
        n=int(hit.sum())
        rows.append((h,lo,hi,minc,minr,cb,n,n/len(targets)))

    res=pd.DataFrame(rows,columns=["Horizon_hours","Min_outflow_ratio","Max_outflow_ratio",
        "Min_outgoing_count","Min_receivers","Require_cross_border","Deposit_Send_hits","Deposit_Send_recall"])
    # Pareto-like ordering: prioritize recall, then fewer/stricter-looking rules for review.
    res=res.sort_values(["Deposit_Send_recall","Require_cross_border","Min_outgoing_count","Min_receivers"],
                        ascending=[False,False,False,False]).reset_index(drop=True)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    res.to_csv(out/"receiver_flow_grid.csv",index=False)
    print("\n=== DEPOSIT-SEND V4 RECEIVER-FLOW SHORTLIST ===")
    print(res.head(25).to_string(index=False))
    print(f"\nSaved: {out/'receiver_flow_grid.csv'}")
    print("IMPORTANT: this grid measures target coverage only. It is for candidate discovery;")
    print("candidate alert volume/precision must be measured before selecting the final rule.")
    print("Development-only. HOLDOUT was not accessed.")

if __name__=="__main__":
    main()
