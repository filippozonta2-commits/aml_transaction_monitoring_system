"""Final DEVELOPMENT-only ablation for the proposed 12-control portfolio.

Measures each control's unique contribution inside the complete portfolio.
No thresholds are changed and HOLDOUT is never accessed.
"""
from pathlib import Path
import argparse, pandas as pd, numpy as np
from evaluate_proposed_12_scenario_portfolio import mismatch_z4_flags, unusual_z4_flags, spark_csv

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
 ap.add_argument("--candidate-flags",default="results/candidate_scenarios/candidate_flags_development.csv")
 ap.add_argument("--geo-flags",default="results/candidate_scenarios/governed_geo_spark")
 ap.add_argument("--frozen-non-ds",default="results/unified_dashboard/materialized_non_deposit_send.csv")
 ap.add_argument("--frozen-ds",default="results/unified_dashboard/deposit_send_flags")
 ap.add_argument("--out",default="results/candidate_scenarios/proposed_12_final_ablation.csv")
 a=ap.parse_args()

 dev=pd.read_csv(a.development); dev["Amount"]=pd.to_numeric(dev["Amount"],errors="coerce")
 c=pd.read_csv(a.candidate_flags).sort_values("development_row_id").reset_index(drop=True)
 g=spark_csv(a.geo_flags)
 if g is None: raise FileNotFoundError("Missing governed geography artifact.")
 g=g.sort_values("development_row_id").reset_index(drop=True)
 if not np.array_equal(c.development_row_id.to_numpy(),g.development_row_id.to_numpy()):
  raise ValueError("candidate/geography row ids do not align")
 for sc in ["SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY"]: c[sc]=g[sc].to_numpy()
 c["SCN_CROSS_BORDER_CURRENCY_MISMATCH_Z4"]=mismatch_z4_flags(dev)
 c["SCN_UNUSUAL_AMOUNT_Z4"]=unusual_z4_flags(dev)

 frozen=["SCN_STRUCTURING","SCN_DEPOSIT_SEND","SCN_FAN_OUT","SCN_FAN_IN","SCN_CASH_WITHDRAWAL","SCN_SMURFING"]
 add=["SCN_CROSS_BORDER_CURRENCY_MISMATCH_Z4","SCN_UNUSUAL_AMOUNT_Z4",
      "SCN_SINGLE_LARGE_TRANSACTION","SCN_GATHER_SCATTER",
      "SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY"]
 f=pd.read_csv(a.frozen_non_ds)
 d=spark_csv(a.frozen_ds)
 if d is None: raise FileNotFoundError("Missing frozen Deposit-Send artifact.")
 f=f[["development_row_id"]+[x for x in frozen if x!="SCN_DEPOSIT_SEND"]].merge(
   d[["development_row_id","SCN_DEPOSIT_SEND"]],on="development_row_id",validate="one_to_one"
 ).sort_values("development_row_id").reset_index(drop=True)
 if not np.array_equal(c.development_row_id.to_numpy(),f.development_row_id.to_numpy()):
  raise ValueError("candidate/frozen row ids do not align")
 for sc in frozen:c[sc]=f[sc].to_numpy()

 y=dev["Is_laundering"].eq(1).reset_index(drop=True); total=int(y.sum())
 controls=frozen+add
 masks={sc:c[sc].eq(1).reset_index(drop=True) for sc in controls}
 full=np.logical_or.reduce([masks[sc].to_numpy() for sc in controls])
 full=pd.Series(full)
 full_alerts=int(full.sum()); full_hits=int((full&y).sum())

 rows=[]
 for sc in controls:
  others=[x for x in controls if x!=sc]
  without=pd.Series(np.logical_or.reduce([masks[x].to_numpy() for x in others]))
  lost=full&~without
  rows.append({
   "scenario":sc,
   "scenario_alerts":int(masks[sc].sum()),
   "scenario_aml_hits":int((masks[sc]&y).sum()),
   "unique_portfolio_alerts":int(lost.sum()),
   "unique_portfolio_aml_hits":int((lost&y).sum()),
   "full_recall_without_scenario":float((without&y).sum()/total),
   "alerts_removed_if_removed":int(full_alerts-without.sum()),
   "aml_hits_lost_if_removed":int(full_hits-(without&y).sum())
  })
 out=pd.DataFrame(rows).sort_values(["aml_hits_lost_if_removed","alerts_removed_if_removed"],ascending=[False,False])
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
 print()
 print("=== PROPOSED 12-SCENARIO FINAL ABLATION — DEVELOPMENT ONLY ===")
 print(f"Full portfolio: {full_alerts:,} alerts | {full_hits:,}/{total:,} AML | recall={full_hits/total:.2%}")
 print(out.to_string(index=False))
 print()
 print("Interpretation: aml_hits_lost_if_removed is the AML coverage uniquely dependent on that control.")
 print("Governed geography controls must also be assessed for policy coverage, not only empirical uniqueness.")
 print("Diagnostic only. No scenario promoted/frozen. No HOLDOUT accessed.")
 print("Saved:",a.out)

if __name__=="__main__": main()
