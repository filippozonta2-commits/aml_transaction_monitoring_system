"""Build dashboard-ready frozen HOLDOUT artifacts.

Presentation only: no model tuning, no threshold selection, no HOLDOUT-driven
feature changes. Creates ranked case queue, KPI summary and scenario summaries.
"""
from pathlib import Path
import argparse
import pandas as pd

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--queue",type=Path,default=Path("results/holdout/investigator_queue.csv"))
 p.add_argument("--features",type=Path,default=Path("results/holdout/case_features_holdout.csv"))
 p.add_argument("--scores",type=Path,default=Path("results/final_holdout/final_holdout_scores.csv"))
 p.add_argument("--alerts",type=Path,default=Path("results/holdout/unified_alert_table_holdout.csv"))
 p.add_argument("--out",type=Path,default=Path("results/dashboard"))
 a=p.parse_args()
 q=pd.read_csv(a.queue); f=pd.read_csv(a.features); s=pd.read_csv(a.scores); al=pd.read_csv(a.alerts)
 d=q.merge(s[["CASE_ID","risk_score"]],on="CASE_ID",validate="one_to_one")
 d=d.merge(f.drop(columns=[c for c in q.columns if c!="CASE_ID" and c in f.columns]),on="CASE_ID",how="left",validate="one_to_one")
 d=d.sort_values("risk_score",ascending=False).reset_index(drop=True); d["priority_rank"]=d.index+1
 d["risk_percentile"]=1-(d.priority_rank-1)/len(d)
 def band(p):
  if p<=100:return "Critical"
  if p<=500:return "High"
  if p<=1000:return "Medium"
  return "Standard"
 d["priority_band"]=d.priority_rank.map(band)
 flags=[c for c in al.columns if c.startswith("SCN_")]
 rows=[]
 for c in flags:
  z=al[c].eq(1); rows.append({"scenario":c,"triggered_transactions":int(z.sum()),"share_of_transactions":float(z.mean())})
 scen=pd.DataFrame(rows).sort_values("triggered_transactions",ascending=False)
 kpi=pd.DataFrame([{"transactions":len(al),"transaction_alerts":int(al["ANY_SCENARIO_ALERT"].sum()),
                    "transaction_alert_rate":float(al["ANY_SCENARIO_ALERT"].mean()),"cases":len(d),
                    "accounts":int(d.account_id.nunique()),"multi_scenario_cases":int(d.multi_scenario_flag.sum()),
                    "median_alerts_per_case":float(d.transaction_alerts.median()),
                    "top100_cases":min(100,len(d)),"top500_cases":min(500,len(d))}])
 a.out.mkdir(parents=True,exist_ok=True)
 d.to_csv(a.out/"ranked_investigator_queue.csv",index=False); scen.to_csv(a.out/"scenario_summary.csv",index=False); kpi.to_csv(a.out/"dashboard_kpis.csv",index=False)
 print("\n=== DASHBOARD DATA MART ==="); print(kpi.to_string(index=False)); print("\n=== SCENARIO VOLUME ==="); print(scen.to_string(index=False)); print(f"\nRanked cases: {len(d):,}"); print("Priority bands are presentation bands based only on frozen rank; they do not change model decisions."); print(f"Saved: {a.out}")
if __name__=="__main__": main()
