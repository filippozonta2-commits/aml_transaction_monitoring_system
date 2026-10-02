"""Evaluate the proposed 12-control AML scenario portfolio on DEVELOPMENT only.

Portfolio = 6 frozen scenarios + 4 empirically useful candidates +
2 governed geography controls.

Excluded from this proposed portfolio:
- SCN_HIGH_TRANSACTION_VELOCITY: requires redesign/tuning
- SCN_BEHAVIORAL_CHANGE: requires redesign/tuning
- zero-trigger second-order graph candidates: remain development candidates

This script evaluates; it does NOT freeze/promote scenarios and never accesses HOLDOUT.
"""
from pathlib import Path
import argparse, pandas as pd, numpy as np

def mismatch_z4_flags(dev):
 """Rebuild the proposed mismatch control using prior sender history only."""
 x=dev.copy()
 dt=pd.to_datetime(x["Date"].astype(str)+" "+x["Time"].astype(str),errors="coerce")
 x["_ts"]=dt
 x["_ord"]=np.arange(len(x))
 x=x.sort_values(["Sender_account","_ts","_ord"])
 grp=x.groupby("Sender_account",sort=False)["Amount"]
 x["_hist_n"]=grp.cumcount()
 x["_hist_avg"]=grp.transform(lambda v:v.shift().expanding().mean())
 x["_hist_sd"]=grp.transform(lambda v:v.shift().expanding().std())
 x["_z"]=(x["Amount"]-x["_hist_avg"])/x["_hist_sd"]
 pt=x["Payment_type"].astype(str).str.strip().str.lower()
 cross=x["Sender_bank_location"].astype(str).str.strip().str.upper().ne(x["Receiver_bank_location"].astype(str).str.strip().str.upper())
 curr=x["Payment_currency"].astype(str).str.strip().str.upper().ne(x["Received_currency"].astype(str).str.strip().str.upper())
 semantic=~pt.isin(["cash withdrawal","cash deposit"])
 x["SCN_CROSS_BORDER_CURRENCY_MISMATCH_Z4"]=(semantic & cross & curr & x["_hist_n"].ge(10) & x["_hist_sd"].gt(0) & x["_z"].ge(4)).astype("int8")
 return x.sort_values("_ord")["SCN_CROSS_BORDER_CURRENCY_MISMATCH_Z4"].to_numpy()

def spark_csv(path):
 p=Path(path); fs=list(p.glob("part-*.csv")) if p.exists() else []
 return pd.concat([pd.read_csv(f) for f in fs],ignore_index=True) if fs else None

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
 ap.add_argument("--candidate-flags",default="results/candidate_scenarios/candidate_flags_development.csv")
 ap.add_argument("--geo-flags",default="results/candidate_scenarios/governed_geo_spark")
 ap.add_argument("--frozen-non-ds",default="results/unified_dashboard/materialized_non_deposit_send.csv")
 ap.add_argument("--frozen-ds",default="results/unified_dashboard/deposit_send_flags")
 ap.add_argument("--out",default="results/candidate_scenarios/proposed_12_scenario_portfolio.csv")
 a=ap.parse_args()

 dev=pd.read_csv(a.development)
 dev["Amount"]=pd.to_numeric(dev["Amount"],errors="coerce")
 c=pd.read_csv(a.candidate_flags).sort_values("development_row_id").reset_index(drop=True)
 g=spark_csv(a.geo_flags)
 if g is None: raise FileNotFoundError("Missing governed geography artifact.")
 g=g.sort_values("development_row_id").reset_index(drop=True)
 if not np.array_equal(c.development_row_id.to_numpy(),g.development_row_id.to_numpy()):
  raise ValueError("candidate/geography row ids do not align")
 for x in ["SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY"]: c[x]=g[x].to_numpy()
 c["SCN_CROSS_BORDER_CURRENCY_MISMATCH_Z4"]=mismatch_z4_flags(dev)

 frozen=["SCN_STRUCTURING","SCN_DEPOSIT_SEND","SCN_FAN_OUT","SCN_FAN_IN","SCN_CASH_WITHDRAWAL","SCN_SMURFING"]
 add=["SCN_CROSS_BORDER_CURRENCY_MISMATCH_Z4","SCN_UNUSUAL_AMOUNT",
      "SCN_SINGLE_LARGE_TRANSACTION","SCN_GATHER_SCATTER",
      "SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY"]
 f=pd.read_csv(a.frozen_non_ds)
 d=spark_csv(a.frozen_ds)
 f=f[["development_row_id"]+[x for x in frozen if x!="SCN_DEPOSIT_SEND"]].merge(
    d[["development_row_id","SCN_DEPOSIT_SEND"]],on="development_row_id",validate="one_to_one"
 ).sort_values("development_row_id").reset_index(drop=True)
 if not np.array_equal(c.development_row_id.to_numpy(),f.development_row_id.to_numpy()):
  raise ValueError("candidate/frozen row ids do not align")
 for x in frozen:c[x]=f[x].to_numpy()

 y=dev.Is_laundering.eq(1); total=int(y.sum()); n=len(y)
 base=c[frozen].eq(1).any(axis=1)
 proposed=base|c[add].eq(1).any(axis=1)

 rows=[]
 current=base.copy()
 for sc in add:
  m=c[sc].eq(1); new=m&~current
  na=int(new.sum()); nh=int((new&y).sum()); current|=m
  rows.append({"scenario":sc,"new_alerts_at_step":na,"new_aml_hits_at_step":nh,
    "marginal_precision":nh/na if na else 0,
    "cumulative_alerts":int(current.sum()),
    "cumulative_alert_rate":float(current.mean()),
    "cumulative_aml_hits":int((current&y).sum()),
    "cumulative_recall":float((current&y).sum()/total)})

 r=pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True,exist_ok=True);r.to_csv(a.out,index=False)
 print()
 print("=== PROPOSED 12-SCENARIO PORTFOLIO — DEVELOPMENT ONLY ===")
 print(f"Transactions: {n:,} | AML positives: {total:,}")
 print(f"6 frozen baseline: {int(base.sum()):,} alerts ({base.mean():.2%}) | {int((base&y).sum()):,} AML | recall={(base&y).sum()/total:.2%}")
 print(r.to_string(index=False))
 print()
 print("=== PORTFOLIO SUMMARY ===")
 print(f"Proposed 12: {int(proposed.sum()):,} alerts ({proposed.mean():.2%} of transactions)")
 print(f"AML hits: {int((proposed&y).sum()):,}/{total:,} | recall={(proposed&y).sum()/total:.2%}")
 print(f"Increment vs frozen: +{int(proposed.sum()-base.sum()):,} alerts | +{int((proposed&y).sum()-(base&y).sum()):,} AML hits")
 print("Excluded for redesign: SCN_HIGH_TRANSACTION_VELOCITY, SCN_BEHAVIORAL_CHANGE")
 print("Zero-trigger graph motifs remain development candidates.")
 print("Cross-border mismatch proposal uses sender-history Z4 + semantic cash guard.")
 print("Governed geography controls retained for policy coverage.")
 print("Evaluation only. No new scenario frozen. No HOLDOUT accessed.")
 print("Saved:",a.out)

if __name__=="__main__":main()
