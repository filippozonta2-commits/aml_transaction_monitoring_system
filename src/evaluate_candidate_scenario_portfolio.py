"""Evaluate candidate scenarios as additions to the frozen scenario portfolio.

DEVELOPMENT ONLY. Transaction-level labels are used strictly for evaluation.
No thresholds are frozen and HOLDOUT is never accessed.
"""
from pathlib import Path
import argparse, pandas as pd, numpy as np

def read_spark_csv(path):
 p=Path(path)
 if not p.exists(): return None
 files=list(p.glob("part-*.csv"))
 return pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else None

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
 ap.add_argument("--candidate-flags",type=Path,default=Path("results/candidate_scenarios/candidate_flags_development.csv"))
 ap.add_argument("--geo-flags",default="results/candidate_scenarios/geo_spark/geo_flags")
 ap.add_argument("--frozen-non-ds",type=Path,default=Path("results/unified_dashboard/materialized_non_deposit_send.csv"))
 ap.add_argument("--frozen-ds",default="results/unified_dashboard/deposit_send_flags")
 ap.add_argument("--out",type=Path,default=Path("results/candidate_scenarios"))
 a=ap.parse_args()
 dev=pd.read_csv(a.development,usecols=["Is_laundering","Laundering_type"])
 cand=pd.read_csv(a.candidate_flags)
 if len(dev)!=len(cand): raise ValueError("candidate/development row mismatch")
 # Add Spark geography output by stable row id when available.
 geo=read_spark_csv(a.geo_flags)
 if geo is not None:
  geo=geo.sort_values("development_row_id").reset_index(drop=True)
  if len(geo)==len(cand):
   for c in ["SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY"]: cand[c]=geo[c].to_numpy()
 # Build the exact six-scenario DEVELOPMENT baseline from the frozen artifacts.
 non=a.frozen_non_ds
 if not non.exists():
  raise FileNotFoundError(f"Missing frozen DEVELOPMENT artifact: {non}. Run: python src/materialize_frozen_scenarios.py")
 fr=pd.read_csv(non)
 ds=read_spark_csv(a.frozen_ds)
 if ds is None:
  raise FileNotFoundError(f"Missing frozen Deposit-Send artifact: {a.frozen_ds}. Run: python src/materialize_deposit_send_spark.py")
 frozen=["SCN_STRUCTURING","SCN_DEPOSIT_SEND","SCN_FAN_OUT","SCN_FAN_IN","SCN_CASH_WITHDRAWAL","SCN_SMURFING"]
 fr=fr[["development_row_id"]+[x for x in frozen if x!="SCN_DEPOSIT_SEND"]]
 ds=ds[["development_row_id","SCN_DEPOSIT_SEND"]]
 fr=fr.merge(ds,on="development_row_id",validate="one_to_one").sort_values("development_row_id").reset_index(drop=True)
 cand=cand.sort_values("development_row_id").reset_index(drop=True)
 if len(fr)!=len(cand) or not np.array_equal(fr.development_row_id.to_numpy(),cand.development_row_id.to_numpy()):
  raise ValueError("Frozen/candidate DEVELOPMENT row ids do not align.")
 for sc in frozen: cand[sc]=fr[sc].to_numpy()
 y=dev.Is_laundering.eq(1); total_pos=int(y.sum()); base=y.mean()
 scenario_cols=[x for x in cand if x.startswith("SCN_")]
 baseline=cand[frozen].eq(1).any(axis=1)
 baseline_pos=y&baseline
 rows=[]
 for sc in scenario_cols:
  if sc in frozen: continue
  m=cand[sc].eq(1); hits=y&m; incremental=hits&~baseline
  typs=dev.loc[hits,"Laundering_type"].dropna().astype(str)
  inc_typs=dev.loc[incremental,"Laundering_type"].dropna().astype(str)
  n=int(m.sum()); hp=int(hits.sum())
  rows.append(dict(scenario=sc,triggered=n,trigger_rate=n/len(dev),aml_hits=hp,
   precision=hp/n if n else 0,recall=hp/total_pos if total_pos else 0,
   lift_vs_base=(hp/n)/base if n and base else 0,overlap_with_frozen=int((m&baseline).sum()),
   incremental_aml_hits=int(incremental.sum()),unique_aml_typologies=typs.nunique(),
   incremental_typologies=inc_typs.nunique()))
 out=pd.DataFrame(rows).sort_values(["incremental_aml_hits","precision"],ascending=False)
 a.out.mkdir(parents=True,exist_ok=True); out.to_csv(a.out/"candidate_portfolio_evaluation.csv",index=False)
 print("\n=== CANDIDATE SCENARIO PORTFOLIO EVALUATION — DEVELOPMENT ONLY ===")
 print(f"Transactions: {len(dev):,} | AML positives: {total_pos:,} | prevalence: {base:.4%}")
 print("Frozen baseline:",", ".join(frozen))
 print(f"Frozen alerted transactions: {int(baseline.sum()):,} | Frozen AML hits: {int(baseline_pos.sum()):,}")
 print(out.to_string(index=False))
 print("\nEvaluation only. No candidate threshold frozen. HOLDOUT was not accessed.")
if __name__=="__main__": main()
