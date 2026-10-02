"""Evaluate selected Deposit-Send V4 receiver-flow candidates on all DEVELOPMENT transactions.

For each candidate, every development transaction is treated as a potential anchor.
The anchor's Receiver_account is followed forward and cumulative outgoing activity
is measured. Reports alert volume, precision, Deposit-Send recall, and lift.
TRAIN is warm-up context only. HOLDOUT is never accessed.
"""
from pathlib import Path
import numpy as np, pandas as pd

COLS=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type","Is_laundering","Laundering_type"]
CANDS=[
 ("A_recall",72,.10,5.0,False),
 ("B_balanced",72,.50,3.0,False),
 ("C_tighter",72,.50,2.0,False),
 ("D_cb_baseline",72,.50,3.0,True),
 ("E_48h",48,.25,3.0,False),
]
def read(p):
 d=pd.read_csv(p,usecols=COLS)
 d["ts"]=pd.to_datetime(pd.to_datetime(d.Date,errors="coerce").dt.strftime("%Y-%m-%d")+" "+d.Time.astype(str),errors="coerce")
 d.Amount=pd.to_numeric(d.Amount,errors="coerce")
 return d.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

def main():
 print("Loading TRAIN warm-up + DEVELOPMENT...")
 tr=read("data/temporal/SAML-D_train.csv"); dev=read("data/temporal/SAML-D_development.csv")
 print(f"Development rows: {len(dev):,} | AML: {int(dev.Is_laundering.sum()):,}")
 print("HOLDOUT is not accessed.")
 maxh=max(x[1] for x in CANDS); start=dev.ts.min()
 ev=pd.concat([tr[tr.ts>=start-pd.Timedelta(hours=maxh)],dev],ignore_index=True).sort_values("ts")
 groups={k:g for k,g in ev.groupby("Sender_account",sort=False)}
 target=(dev.Is_laundering.eq(1)&dev.Laundering_type.eq("Deposit-Send"))
 target_n=int(target.sum()); overall=dev.Is_laundering.mean()
 rows=[]
 # Straightforward implementation; candidate count is intentionally small.
 for name,h,lo,hi,reqcb in CANDS:
  print(f"Evaluating {name}: {h}h ratio {lo}-{hi} cross-border={reqcb}...")
  flags=np.zeros(len(dev),dtype=bool)
  for i,r in enumerate(dev.itertuples()):
   g=groups.get(r.Receiver_account)
   if g is None or pd.isna(r.Amount) or r.Amount<=0: continue
   z=g[(g.ts>r.ts)&(g.ts<=r.ts+pd.Timedelta(hours=h))]
   if z.empty: continue
   ratio=z.Amount.sum()/r.Amount
   flags[i]=(lo<=ratio<=hi) and ((z.Payment_type=="Cross-border").any() if reqcb else True)
  trig=int(flags.sum()); aml=int(dev.loc[flags,"Is_laundering"].sum())
  ds=int((flags & target.to_numpy()).sum())
  precision=aml/trig if trig else 0.; recall=ds/target_n if target_n else 0.
  rows.append((name,h,lo,hi,reqcb,trig,trig/len(dev),aml,precision,precision/overall if overall else np.nan,ds,recall))
 res=pd.DataFrame(rows,columns=["Candidate","Horizon_hours","Min_ratio","Max_ratio","Require_cross_border",
  "Triggered_transactions","Trigger_rate","AML_cases","Precision","Lift_vs_overall_AML_rate","Deposit_Send_hits","Deposit_Send_recall"])
 res=res.sort_values(["Deposit_Send_recall","Precision"],ascending=False)
 out=Path("results/deposit_send_v4_evaluation"); out.mkdir(parents=True,exist_ok=True)
 res.to_csv(out/"candidate_evaluation.csv",index=False)
 print("\n=== DEPOSIT-SEND V4 CANDIDATE EVALUATION ===")
 print(res.to_string(index=False))
 print(f"\nOverall development AML rate: {overall:.4%}")
 print(f"Deposit-Send targets: {target_n}")
 print(f"Saved: {out/'candidate_evaluation.csv'}")
 print("Development-only. HOLDOUT was not accessed.")
if __name__=="__main__": main()
