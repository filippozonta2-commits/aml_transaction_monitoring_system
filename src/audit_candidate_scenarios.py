"""Audit 12 new scenario candidates on DEVELOPMENT only.
No threshold selection and no HOLDOUT access.
"""
from pathlib import Path
import argparse, pandas as pd

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
 p.add_argument("--flags",type=Path,default=Path("results/candidate_scenarios/candidate_flags_development.csv"))
 p.add_argument("--out",type=Path,default=Path("results/candidate_scenarios"))
 a=p.parse_args()
 x=pd.read_csv(a.development,usecols=["Is_laundering","Laundering_type"])
 f=pd.read_csv(a.flags)
 if len(x)!=len(f): raise ValueError(f"Row mismatch: development={len(x):,}, flags={len(f):,}")
 rows=[]
 base=float((x.Is_laundering==1).mean())
 for sc in [c for c in f.columns if c.startswith("SCN_")]:
  m=f[sc].eq(1); n=int(m.sum()); pos=int(((x.Is_laundering==1)&m).sum())
  precision=pos/n if n else 0.0
  rows.append({"scenario":sc,"triggered":n,"trigger_rate":n/len(x),"aml_hits":pos,
   "precision":precision,"lift_vs_base":precision/base if base and n else 0.0,
   "unique_aml_typologies":x.loc[m&(x.Is_laundering==1),"Laundering_type"].nunique()})
 out=pd.DataFrame(rows).sort_values(["precision","aml_hits"],ascending=False)
 a.out.mkdir(parents=True,exist_ok=True); out.to_csv(a.out/"candidate_label_audit.csv",index=False)
 print("\n=== CANDIDATE LABEL AUDIT — DEVELOPMENT ONLY ===")
 print(f"Rows: {len(x):,} | AML prevalence: {base:.4%}")
 print(out.to_string(index=False))
 print("\nDiagnostic only. Do not freeze thresholds from this table alone.")
 print("HOLDOUT was not accessed.")
if __name__=="__main__": main()
