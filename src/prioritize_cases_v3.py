"""Prioritize aggregated V3 investigation cases using transparent operational signals.

No Is_laundering labels are used. This is downstream queue prioritization only and
does not alter frozen V3 detection, thresholds, or HOLDOUT validation metrics.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

SCENARIO_POINTS={
 "SCN_SANCTIONED_GEOGRAPHY":40,
 "SCN_HIGH_RISK_GEOGRAPHY":25,
 "SCN_CASH_WITHDRAWAL":20,
 "SCN_SMURFING":20,
 "SCN_FAN_IN":15,
 "SCN_FAN_OUT":15,
 "SCN_STRUCTURING":15,
 "SCN_GATHER_SCATTER":15,
 "SCN_DEPOSIT_SEND":10,
 "SCN_SINGLE_LARGE_TRANSACTION":10,
 "SCN_UNUSUAL_AMOUNT_Z4":10,
}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--cases",default="results/case_management_v3/aggregated_cases/case.csv")
 p.add_argument("--case-alert",default="results/case_management_v3/aggregated_cases/case_alert.csv")
 p.add_argument("--alerts",default="results/case_management_v3/aggregated/alert.csv")
 p.add_argument("--outdir",default="results/case_management_v3/prioritized")
 a=p.parse_args()

 cases=pd.read_csv(a.cases)
 ca=pd.read_csv(a.case_alert)
 alerts=pd.read_csv(a.alerts)
 if "Is_laundering" in cases.columns or "Is_laundering" in alerts.columns:
  raise ValueError("Label leakage guard: Is_laundering must not enter prioritization.")

 detail=ca[["case_id","alert_id"]].merge(
  alerts[["alert_id","scenario_id","transaction_count","alert_amount","policy_flag"]],
  on="alert_id",how="left",validate="many_to_one")
 detail["scenario_points"]=detail["scenario_id"].map(SCENARIO_POINTS)
 if detail["scenario_points"].isna().any():
  raise ValueError("Unmapped scenario in prioritization.")

 # Score components are intentionally simple, bounded, and auditable.
 base=detail.groupby("case_id")["scenario_points"].max().rename("scenario_severity_score")
 policy=detail.groupby("case_id")["policy_flag"].max().astype(bool).rename("_policy")
 out=cases.merge(base,on="case_id",how="left").merge(policy,on="case_id",how="left")

 out["multi_scenario_score"]=np.select(
  [out.scenario_count.ge(4),out.scenario_count.eq(3),out.scenario_count.eq(2)],
  [20,15,10],default=0)
 out["alert_density_score"]=np.select(
  [out.alert_count.ge(6),out.alert_count.ge(4),out.alert_count.ge(2)],
  [15,10,5],default=0)
 out["transaction_volume_score"]=np.select(
  [out.transaction_count.ge(100),out.transaction_count.ge(25),out.transaction_count.ge(5)],
  [15,10,5],default=0)

 out["risk_score"]=(out.scenario_severity_score+out.multi_scenario_score+
                    out.alert_density_score+out.transaction_volume_score).clip(upper=100).astype(int)
 # Policy controls are never allowed to fall into LOW operational priority.
 out["queue_priority"]=pd.cut(out.risk_score,[-1,24,49,100],labels=["LOW","MEDIUM","HIGH"]).astype(str)
 out.loc[out["_policy"] & out.queue_priority.eq("LOW"),"queue_priority"]="MEDIUM"

 out["priority_reason"]=out.apply(
  lambda r:(f"severity={int(r.scenario_severity_score)}; multi_scenario={int(r.multi_scenario_score)}; "
            f"alert_density={int(r.alert_density_score)}; transaction_volume={int(r.transaction_volume_score)}"
            + ("; policy_control=true" if bool(r._policy) else "")),axis=1)
 out=out.drop(columns=["_policy"])

 order={"HIGH":0,"MEDIUM":1,"LOW":2}
 out["_q"]=out.queue_priority.map(order)
 out=out.sort_values(["_q","risk_score","case_created_at"],ascending=[True,False,True]).drop(columns="_q")
 out["queue_rank"]=np.arange(1,len(out)+1)

 od=Path(a.outdir);od.mkdir(parents=True,exist_ok=True)
 out.to_csv(od/"case_priority_queue.csv",index=False)
 detail.to_csv(od/"case_priority_evidence.csv",index=False)

 summary=(out.groupby("queue_priority",as_index=False)
          .agg(cases=("case_id","count"),
               avg_score=("risk_score","mean"),
               avg_alerts=("alert_count","mean"),
               avg_transactions=("transaction_count","mean")))
 summary["share"]=summary.cases/len(out)
 summary.to_csv(od/"priority_summary.csv",index=False)

 manifest={
  "method":"transparent rule-based operational prioritization",
  "label_leakage":"Is_laundering not used",
  "scenario_points":SCENARIO_POINTS,
  "score_components":{
   "multi_scenario":{"2":10,"3":15,"4+":20},
   "alert_density":{"2-3":5,"4-5":10,"6+":15},
   "transaction_volume":{"5-24":5,"25-99":10,"100+":15}},
  "bands":{"LOW":"0-24","MEDIUM":"25-49","HIGH":"50-100"},
  "policy_floor":"Policy-linked cases cannot be LOW",
  "note":"Operational prioritization only; V3 detection and HOLDOUT validation unchanged."
 }
 (od/"prioritization_manifest.json").write_text(json.dumps(manifest,indent=2))

 print("=== V3 CASE PRIORITIZATION ===")
 print(f"Cases prioritized: {len(out):,}")
 print("\nQueue distribution:")
 print(summary.to_string(index=False))
 print("\nRisk score distribution:")
 print(out.risk_score.describe(percentiles=[.5,.75,.9,.95,.99]).to_string())
 print("\nTop 15 queue:")
 print(out[["queue_rank","case_id","subject_id","risk_score","queue_priority",
            "alert_count","transaction_count","scenario_count","policy_flag","scenarios"]].head(15).to_string(index=False))
 print("\nNo AML labels used. Frozen V3 detection/validation unchanged.")
 print("Saved:",od)

if __name__=="__main__": main()
