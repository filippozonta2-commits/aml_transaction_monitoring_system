"""Threshold diagnostics for broad candidate scenarios on DEVELOPMENT only.
Ranks compact alternatives; does not freeze any scenario and never reads HOLDOUT.
"""
from pathlib import Path
import argparse, numpy as np, pandas as pd

def metric(name,m,y,rows):
 n=int(m.sum()); hits=int((m&y).sum()); p=hits/n if n else 0
 rows.append((name,n,n/len(y),hits,p))

def main():
 p=argparse.ArgumentParser(); p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv")); p.add_argument("--out",type=Path,default=Path("results/candidate_scenarios")); a=p.parse_args()
 cols=["Date","Time","Sender_account","Receiver_account","Amount","Payment_currency","Received_currency","Sender_bank_location","Receiver_bank_location","Is_laundering"]
 x=pd.read_csv(a.development,usecols=cols); y=x.Is_laundering.eq(1); x["ts"]=pd.to_datetime(pd.to_datetime(x.Date,errors="coerce").dt.strftime("%Y-%m-%d")+" "+x.Time.astype(str),errors="coerce"); x["day"]=x.ts.dt.floor("D")
 x["cross"]=x.Sender_bank_location.ne(x.Receiver_bank_location); x["fx"]=x.Payment_currency.ne(x.Received_currency)
 rows=[]
 # FX mismatch: add amount tail to avoid flagging routine FX traffic.
 cb=x.cross&x.fx
 for q in [.50,.75,.90,.95,.975,.99]:
  cut=x.loc[cb,"Amount"].quantile(q); metric(f"FX_MISMATCH_amount_q{q}",cb&x.Amount.ge(cut),y,rows)
 # Velocity: use account-day count quantiles rather than current transaction-level threshold.
 d=x.groupby(["day","Sender_account"]).size().rename("n").reset_index(); z=x.merge(d,on=["day","Sender_account"],how="left")
 for q in [.95,.975,.99,.995,.999]:
  cut=d.n.quantile(q); metric(f"VELOCITY_daily_count_q{q}_cut{cut:.0f}",z.n.ge(cut),y,rows)
 # Behavioral: account-day count/value relative to prior expanding history.
 d=x.groupby(["day","Sender_account"]).agg(n=("Amount","size"),amt=("Amount","sum")).reset_index().sort_values(["Sender_account","day"])
 d["pn"]=d.groupby("Sender_account").n.transform(lambda s:s.shift().expanding(7).mean()); d["pa"]=d.groupby("Sender_account").amt.transform(lambda s:s.shift().expanding(7).mean())
 for nmult,amult in [(3,4),(4,5),(5,6),(6,8),(8,10)]:
  d["hit"]=d.pn.notna()&((d.n>=nmult*d.pn)&(d.amt>=amult*d.pa))
  zz=x.merge(d[["day","Sender_account","hit"]],on=["day","Sender_account"],how="left")
  metric(f"BEHAVIOR_change_n{nmult}x_amt{amult}x",zz.hit.fillna(False),y,rows)
 out=pd.DataFrame(rows,columns=["candidate","triggered","trigger_rate","aml_hits","precision"])
 base=y.mean(); out["lift_vs_base"]=out.precision/base; out=out.sort_values(["precision","aml_hits"],ascending=False)
 a.out.mkdir(parents=True,exist_ok=True); out.to_csv(a.out/"candidate_threshold_benchmark.csv",index=False)
 print("\n=== BROAD-SCENARIO THRESHOLD BENCHMARK — DEVELOPMENT ONLY ==="); print(out.to_string(index=False))
 print("\nNo candidate frozen. HOLDOUT was not accessed.")
if __name__=="__main__": main()
