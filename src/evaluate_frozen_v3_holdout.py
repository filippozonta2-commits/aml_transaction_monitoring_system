"""One-shot out-of-sample evaluation of the frozen V3 11-control portfolio."""
from pathlib import Path
import argparse,pandas as pd,numpy as np
def spark_csv(p):
 fs=list(Path(p).glob("part-*.csv"));return pd.concat([pd.read_csv(f) for f in fs],ignore_index=True) if fs else None
def main():
 p=argparse.ArgumentParser();p.add_argument("--base",default="results/holdout_v3/non_deposit_send.csv");p.add_argument("--ds",default="results/holdout_v3/deposit_send");p.add_argument("--geo",default="results/holdout_v3/governed_geo");p.add_argument("--out",default="results/holdout_v3/final_metrics.csv");a=p.parse_args()
 x=pd.read_csv(a.base).sort_values("holdout_row_id").reset_index(drop=True);d=spark_csv(a.ds);g=spark_csv(a.geo)
 if d is None or g is None: raise FileNotFoundError("Run all HOLDOUT materializers first.")
 for z in [d,g]:
  z.sort_values("holdout_row_id",inplace=True);z.reset_index(drop=True,inplace=True)
  if not np.array_equal(x.holdout_row_id.to_numpy(),z.holdout_row_id.to_numpy()):raise ValueError("HOLDOUT row ids do not align.")
 x=x.merge(d,on="holdout_row_id",validate="one_to_one").merge(g,on="holdout_row_id",validate="one_to_one")
 controls=["SCN_STRUCTURING","SCN_DEPOSIT_SEND","SCN_FAN_OUT","SCN_FAN_IN","SCN_CASH_WITHDRAWAL","SCN_SMURFING","SCN_UNUSUAL_AMOUNT_Z4","SCN_SINGLE_LARGE_TRANSACTION","SCN_GATHER_SCATTER","SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY"]
 y=x.Is_laundering.eq(1);total=int(y.sum());rows=[]
 for sc in controls:
  m=x[sc].eq(1);rows.append({"scenario":sc,"alerts":int(m.sum()),"alert_rate":float(m.mean()),"aml_hits":int((m&y).sum()),"precision":float((m&y).sum()/m.sum()) if m.sum() else 0,"recall":float((m&y).sum()/total) if total else 0})
 anym=x[controls].eq(1).any(axis=1);hits=int((anym&y).sum())
 rows.append({"scenario":"PORTFOLIO_V3","alerts":int(anym.sum()),"alert_rate":float(anym.mean()),"aml_hits":hits,"precision":float(hits/anym.sum()) if anym.sum() else 0,"recall":float(hits/total) if total else 0})
 out=pd.DataFrame(rows);Path(a.out).parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.out,index=False)
 print("=== FROZEN V3 — HOLDOUT OUT-OF-SAMPLE EVALUATION ===");print(f"Transactions: {len(x):,} | AML positives: {total:,} | prevalence={total/len(x):.4%}");print(out.to_string(index=False))
 print("\nDEVELOPMENT reference: 131,921 alerts (6.94%) | 1,189/1,986 AML | recall=59.87%")
 print("HOLDOUT has now been accessed. Results are validation only; DO NOT tune V3 on these results.");print("Saved:",a.out)
if __name__=="__main__":main()
