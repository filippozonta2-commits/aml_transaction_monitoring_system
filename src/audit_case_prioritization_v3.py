"""Audit V3 case-priority composition without AML labels.

Governance QA for the operational queue only. No tuning of frozen V3 detection,
thresholds, or HOLDOUT validation.
"""
from pathlib import Path
import argparse, json
import pandas as pd

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--queue",default="results/case_management_v3/prioritized/case_priority_queue.csv")
 p.add_argument("--evidence",default="results/case_management_v3/prioritized/case_priority_evidence.csv")
 p.add_argument("--outdir",default="results/case_management_v3/prioritized/audit")
 a=p.parse_args()

 q=pd.read_csv(a.queue)
 ev=pd.read_csv(a.evidence)
 forbidden={"Is_laundering","Laundering_type"}
 if forbidden & set(q.columns) or forbidden & set(ev.columns):
  raise ValueError("Label leakage guard failed.")

 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)

 # Queue-level profile.
 profile=(q.groupby("queue_priority",as_index=False)
   .agg(cases=("case_id","nunique"),
        avg_score=("risk_score","mean"),
        median_score=("risk_score","median"),
        avg_alerts=("alert_count","mean"),
        avg_transactions=("transaction_count","mean"),
        avg_scenarios=("scenario_count","mean"),
        policy_cases=("policy_flag","sum")))
 profile["share"]=profile["cases"]/q["case_id"].nunique()
 profile["policy_share"]=profile["policy_cases"]/profile["cases"]
 profile.to_csv(out/"queue_profile.csv",index=False)

 # Scenario representation by priority band.
 qe=q[["case_id","queue_priority","risk_score"]]
 se=ev[["case_id","scenario_id"]].drop_duplicates().merge(qe,on="case_id",how="inner")
 scen=(se.groupby(["queue_priority","scenario_id"],as_index=False)
       .agg(cases=("case_id","nunique"),avg_case_score=("risk_score","mean")))
 totals=q.groupby("queue_priority")["case_id"].nunique()
 scen["share_of_band"]=scen.apply(lambda r:r["cases"]/totals[r["queue_priority"]],axis=1)
 scen.to_csv(out/"scenario_by_priority.csv",index=False)

 # Explain which score components are present in HIGH.
 components=["scenario_severity_score","multi_scenario_score",
             "alert_density_score","transaction_volume_score"]
 high=q[q.queue_priority.eq("HIGH")].copy()
 rows=[]
 for c in components:
  rows.append({"component":c,"high_cases_with_component":int(high[c].gt(0).sum()),
               "share_of_high":float(high[c].gt(0).mean()),
               "avg_points_when_present":float(high.loc[high[c].gt(0),c].mean())})
 pd.DataFrame(rows).to_csv(out/"high_component_audit.csv",index=False)

 # Policy isolation: cases containing only one policy scenario versus policy + other signals.
 policy_scen={"SCN_SANCTIONED_GEOGRAPHY","SCN_HIGH_RISK_GEOGRAPHY"}
 case_scen=ev.groupby("case_id")["scenario_id"].agg(lambda s:set(s))
 q["_scenario_set"]=q.case_id.map(case_scen)
 q["policy_only_single_scenario"]=q["_scenario_set"].apply(
   lambda s: isinstance(s,set) and len(s)==1 and next(iter(s)) in policy_scen)
 q["policy_plus_other_scenarios"]=q["_scenario_set"].apply(
   lambda s: isinstance(s,set) and bool(s & policy_scen) and len(s)>1)

 policy_audit=(q.groupby("queue_priority",as_index=False)
  .agg(policy_cases=("policy_flag","sum"),
       policy_only_single_scenario=("policy_only_single_scenario","sum"),
       policy_plus_other_scenarios=("policy_plus_other_scenarios","sum")))
 policy_audit.to_csv(out/"policy_priority_audit.csv",index=False)

 manifest={
  "cases":int(q.case_id.nunique()),
  "high_cases":int(q.queue_priority.eq("HIGH").sum()),
  "medium_cases":int(q.queue_priority.eq("MEDIUM").sum()),
  "low_cases":int(q.queue_priority.eq("LOW").sum()),
  "high_policy_share":float(high.policy_flag.astype(bool).mean()),
  "high_multi_scenario_share":float(high.scenario_count.gt(1).mean()),
  "high_100plus_transaction_share":float(high.transaction_count.ge(100).mean()),
  "label_leakage":"No AML outcome labels used.",
  "purpose":"Governance audit of operational priority composition; not model/detection tuning."
 }
 (out/"priority_audit_manifest.json").write_text(json.dumps(manifest,indent=2))

 print("=== V3 CASE PRIORITY AUDIT ===")
 print("\nQueue profile:")
 print(profile.to_string(index=False))
 print("\nHIGH component contribution:")
 print(pd.DataFrame(rows).to_string(index=False))
 print("\nPolicy composition:")
 print(policy_audit.to_string(index=False))
 print("\nTop scenario representation by band:")
 for band in ["HIGH","MEDIUM","LOW"]:
  z=scen[scen.queue_priority.eq(band)].sort_values("share_of_band",ascending=False).head(8)
  print(f"\n{band}:")
  print(z[["scenario_id","cases","share_of_band","avg_case_score"]].to_string(index=False))
 print("\nGovernance audit only. No AML labels used; frozen V3 unchanged.")
 print("Saved:",out)

if __name__=="__main__": main()
