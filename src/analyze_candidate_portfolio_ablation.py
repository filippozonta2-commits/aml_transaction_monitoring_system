"""Candidate scenario rationalization / ablation analysis — DEVELOPMENT only.

Measures each candidate's standalone value, value beyond the frozen baseline,
and leave-one-out contribution inside the full candidate portfolio.
No automatic promotion decision. No HOLDOUT access.
"""
from pathlib import Path
import argparse, pandas as pd, numpy as np

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
 ap.add_argument("--out",default="results/candidate_scenarios/candidate_ablation.csv")
 a=ap.parse_args()

 dev=pd.read_csv(a.development,usecols=["Is_laundering","Laundering_type"])
 c=pd.read_csv(a.candidate_flags).sort_values("development_row_id").reset_index(drop=True)
 g=spark_csv(a.geo_flags)
 if g is None: raise FileNotFoundError("Missing governed geography artifact.")
 g=g.sort_values("development_row_id").reset_index(drop=True)
 if not np.array_equal(c.development_row_id.to_numpy(),g.development_row_id.to_numpy()):
  raise ValueError("candidate/geography row ids do not align")
 for x in ["SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY"]: c[x]=g[x].to_numpy()

 frozen=["SCN_STRUCTURING","SCN_DEPOSIT_SEND","SCN_FAN_OUT","SCN_FAN_IN","SCN_CASH_WITHDRAWAL","SCN_SMURFING"]
 f=pd.read_csv(a.frozen_non_ds)
 d=spark_csv(a.frozen_ds)
 f=f[["development_row_id"]+[x for x in frozen if x!="SCN_DEPOSIT_SEND"]].merge(
   d[["development_row_id","SCN_DEPOSIT_SEND"]],on="development_row_id",validate="one_to_one"
 ).sort_values("development_row_id").reset_index(drop=True)
 if not np.array_equal(c.development_row_id.to_numpy(),f.development_row_id.to_numpy()):
  raise ValueError("candidate/frozen row ids do not align")
 for x in frozen: c[x]=f[x].to_numpy()

 y=dev.Is_laundering.eq(1); total=int(y.sum())
 base=c[frozen].eq(1).any(axis=1)
 candidates=[x for x in c if x.startswith("SCN_") and x not in frozen]
 full=base|c[candidates].eq(1).any(axis=1)

 rows=[]
 for sc in candidates:
  m=c[sc].eq(1)
  inc=m&~base
  others=[x for x in candidates if x!=sc]
  without=base|c[others].eq(1).any(axis=1)
  unique=m&~without
  rows.append({
   "scenario":sc,
   "standalone_alerts":int(m.sum()),
   "standalone_aml_hits":int((m&y).sum()),
   "incremental_vs_frozen_alerts":int(inc.sum()),
   "incremental_vs_frozen_aml_hits":int((inc&y).sum()),
   "incremental_vs_frozen_precision":float((inc&y).sum()/inc.sum()) if inc.sum() else 0,
   "unique_portfolio_alerts":int(unique.sum()),
   "unique_portfolio_aml_hits":int((unique&y).sum()),
   "aml_hits_lost_if_removed":int((full&y).sum()-(without&y).sum()),
   "alerts_removed_if_removed":int(full.sum()-without.sum()),
   "full_recall_without_scenario":float((without&y).sum()/total) if total else 0
  })

 r=pd.DataFrame(rows).sort_values(
   ["aml_hits_lost_if_removed","incremental_vs_frozen_aml_hits","alerts_removed_if_removed"],
   ascending=[False,False,True])
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); r.to_csv(a.out,index=False)

 print("\n=== CANDIDATE PORTFOLIO ABLATION — DEVELOPMENT ONLY ===")
 print(f"Frozen: {int(base.sum()):,} alerts | {int((base&y).sum()):,}/{total:,} AML | recall={(base&y).sum()/total:.2%}")
 print(f"Full candidate portfolio: {int(full.sum()):,} alerts | {int((full&y).sum()):,}/{total:,} AML | recall={(full&y).sum()/total:.2%}")
 print(r.to_string(index=False))
 print("\nInterpretation: aml_hits_lost_if_removed = AML positives uniquely dependent on that candidate inside the full portfolio.")
 print("Governed geography controls should be assessed separately for policy coverage; empirical ablation does not override policy.")
 print("Saved:",a.out)
 print("Diagnostic only. No scenario automatically promoted/frozen. No HOLDOUT accessed.")

if __name__=="__main__": main()
