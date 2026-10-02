"""Evaluate cumulative/marginal DEVELOPMENT coverage for candidate AML scenarios.

Diagnostic governance artifact only. No HOLDOUT access and no automatic freezing.
Policy-driven geography scenarios remain included regardless of empirical ordering.
"""
from pathlib import Path
import argparse, pandas as pd, numpy as np

def read_spark_csv(path):
 p=Path(path); files=list(p.glob("part-*.csv")) if p.exists() else []
 return pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else None

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--development",default="data/temporal/SAML-D_development.csv")
 p.add_argument("--candidate-flags",default="results/candidate_scenarios/candidate_flags_development.csv")
 p.add_argument("--geo-flags",default="results/candidate_scenarios/governed_geo_spark")
 p.add_argument("--frozen-non-ds",default="results/unified_dashboard/materialized_non_deposit_send.csv")
 p.add_argument("--frozen-ds",default="results/unified_dashboard/deposit_send_flags")
 p.add_argument("--out",default="results/candidate_scenarios")
 a=p.parse_args()

 dev=pd.read_csv(a.development,usecols=["Is_laundering","Laundering_type"])
 cand=pd.read_csv(a.candidate_flags).sort_values("development_row_id").reset_index(drop=True)
 geo=read_spark_csv(a.geo_flags)
 if geo is None: raise FileNotFoundError("Run governed geography materializer first.")
 geo=geo.sort_values("development_row_id").reset_index(drop=True)
 if not np.array_equal(cand.development_row_id.to_numpy(),geo.development_row_id.to_numpy()):
  raise ValueError("candidate/geography row ids do not align")
 for c in ["SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY"]: cand[c]=geo[c].to_numpy()

 frozen=["SCN_STRUCTURING","SCN_DEPOSIT_SEND","SCN_FAN_OUT","SCN_FAN_IN","SCN_CASH_WITHDRAWAL","SCN_SMURFING"]
 fr=pd.read_csv(a.frozen_non_ds)
 ds=read_spark_csv(a.frozen_ds)
 fr=fr[["development_row_id"]+[x for x in frozen if x!="SCN_DEPOSIT_SEND"]].merge(
    ds[["development_row_id","SCN_DEPOSIT_SEND"]],on="development_row_id",validate="one_to_one"
 ).sort_values("development_row_id").reset_index(drop=True)
 if not np.array_equal(cand.development_row_id.to_numpy(),fr.development_row_id.to_numpy()):
  raise ValueError("candidate/frozen row ids do not align")
 for c in frozen: cand[c]=fr[c].to_numpy()

 candidates=[c for c in cand.columns if c.startswith("SCN_") and c not in frozen]
 y=dev.Is_laundering.eq(1); total=int(y.sum())
 base=cand[frozen].eq(1).any(axis=1)
 remaining=set(candidates); selected=[]; current=base.copy(); rows=[]

 # Greedy ordering maximizes new AML hits; tie-break by fewer new alerts.
 while remaining:
  scored=[]
  for sc in remaining:
   m=cand[sc].eq(1); new=m&~current
   scored.append((int((new&y).sum()),-int(new.sum()),sc))
  _,_,best=max(scored)
  m=cand[best].eq(1); new=m&~current
  new_alerts=int(new.sum()); new_hits=int((new&y).sum())
  current=current|m; selected.append(best); remaining.remove(best)
  rows.append(dict(step=len(selected),scenario=best,new_alerts=new_alerts,new_aml_hits=new_hits,
   marginal_precision=new_hits/new_alerts if new_alerts else 0,
   cumulative_alerts=int(current.sum()),cumulative_aml_hits=int((current&y).sum()),
   cumulative_aml_recall=int((current&y).sum())/total if total else 0))

 out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(out/"candidate_portfolio_greedy.csv",index=False)

 # Pairwise Jaccard + raw overlap on transaction triggers.
 j=pd.DataFrame(index=candidates,columns=candidates,dtype=float)
 ov=pd.DataFrame(index=candidates,columns=candidates,dtype=int)
 for x in candidates:
  mx=cand[x].eq(1)
  for z in candidates:
   mz=cand[z].eq(1); inter=int((mx&mz).sum()); union=int((mx|mz).sum())
   ov.loc[x,z]=inter; j.loc[x,z]=inter/union if union else 0
 ov.to_csv(out/"candidate_overlap_counts.csv"); j.to_csv(out/"candidate_overlap_jaccard.csv")

 print("\n=== CANDIDATE PORTFOLIO — GREEDY INCREMENTAL DEVELOPMENT COVERAGE ===")
 print(f"Frozen baseline: alerts={int(base.sum()):,} AML={int((base&y).sum()):,}/{total:,} recall={(base&y).sum()/total:.2%}")
 print(pd.DataFrame(rows).to_string(index=False))
 print("\nSaved overlap matrices and greedy portfolio:",out)
 print("Diagnostic only: greedy order is NOT an automatic promotion/ranking policy. No HOLDOUT accessed.")

if __name__=="__main__": main()
