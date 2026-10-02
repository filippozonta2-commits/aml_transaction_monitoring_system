"""Reconcile materialized Smurfing alerts against the exact frozen definition.

The materializer currently matches frozen target recall but has extra non-AML
alerts. This diagnostic tests the likely omitted account-window constraints
without changing thresholds or touching HOLDOUT.
"""
from pathlib import Path
import argparse, numpy as np, pandas as pd

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
 p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
 p.add_argument("--output-dir",type=Path,default=Path("results/smurfing_reconciliation"))
 a=p.parse_args()
 use=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
      "Payment_currency","Received_currency","Sender_bank_location","Receiver_bank_location",
      "Is_laundering","Laundering_type"]
 def load(path,period):
  d=pd.read_csv(path,usecols=use)
  d["ts"]=pd.to_datetime(pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")+" "+d["Time"].astype(str),errors="coerce")
  d=d.dropna(subset=["ts"]); d["Period"]=period
  d["cross"]=(d["Sender_bank_location"]!=d["Receiver_bank_location"]).astype("int8")
  d["fx"]=(d["Payment_currency"]!=d["Received_currency"]).astype("int8")
  return d
 tr=load(a.train,"train"); dev=load(a.development,"development")
 start=dev.ts.min(); warm=tr[tr.ts.ge(start-pd.Timedelta(days=60))]
 h=pd.concat([warm,dev],ignore_index=True).sort_values("ts")
 h=h[h.Payment_type.eq("Cash Deposit")].copy()
 origin=start
 h["window_id"]=np.floor((h.ts-origin).dt.total_seconds()/86400/45).astype("int32")
 g=h.groupby(["window_id","Sender_account"],observed=True).agg(
   tx_count=("Amount","size"),aggregate_amount=("Amount","sum"),median_amount=("Amount","median"),
   unique_receivers=("Receiver_account","nunique"),cross_border_rate=("cross","mean"),
   currency_mismatch_rate=("fx","mean")).reset_index()
 x=h.merge(g,on=["window_id","Sender_account"],how="left")
 base=x.tx_count.ge(3)&x.median_amount.lt(4000)&x.aggregate_amount.ge(10000)
 variants={
  "frozen_anchor_base":base,
  "materializer_base":base,
  "plus_unique_receivers_le3":base&x.unique_receivers.le(3),
  "plus_geo_fx_le15":base&x.cross_border_rate.le(.15)&x.currency_mismatch_rate.le(.15),
  "full_legacy_constraints":base&x.unique_receivers.le(3)&x.cross_border_rate.le(.15)&x.currency_mismatch_rate.le(.15),
 }
 rows=[]
 xd=x[x.Period.eq("development")]
 for name,mask in variants.items():
  m=mask.loc[xd.index]; aml=xd.Is_laundering.eq(1); target=aml&xd.Laundering_type.eq("Smurfing")
  trig=int(m.sum()); tp=int((m&aml).sum()); hits=int((m&target).sum())
  rows.append({"Variant":name,"Triggered":trig,"AML_cases":tp,"Precision":tp/trig if trig else np.nan,
               "Smurfing_hits":hits,"Smurfing_recall":hits/target.sum()})
 out=pd.DataFrame(rows)
 a.output_dir.mkdir(parents=True,exist_ok=True); out.to_csv(a.output_dir/"reconciliation.csv",index=False)
 print("\n=== SMURFING RECONCILIATION ==="); print(out.to_string(index=False))
 print("\nFrozen target: Triggered=2204 | AML_cases=199 | Smurfing_hits=199 | recall=0.925581")
 print("No thresholds were tuned. Development-only. HOLDOUT was not accessed.")
if __name__=="__main__": main()
