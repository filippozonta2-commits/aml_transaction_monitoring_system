"""Aggregate frozen V3 transaction-level scenario alerts into investigation alerts.

Downstream operationalization only. Frozen V3 scenario flags, thresholds and HOLDOUT
validation are unchanged.

Aggregation grain:
    primary_account_id x scenario_id x scenario-specific inactivity session

The session gap is aligned to the detection horizon where known from frozen V3:
- STRUCTURING: 10d
- DEPOSIT_SEND: 3d (72h)
- FAN_OUT: 21d
- FAN_IN: 10d
- CASH_WITHDRAWAL: 7d
- SMURFING: 45d
- UNUSUAL_AMOUNT_Z4: 1d
- SINGLE_LARGE_TRANSACTION: 1d
- GATHER_SCATTER: 1d
- HIGH_RISK_GEOGRAPHY: 30d operational consolidation
- SANCTIONED_GEOGRAPHY: 30d operational consolidation

Geography windows are alert-workflow consolidation choices, not detection thresholds.
"""
from pathlib import Path
import argparse, hashlib, json
import pandas as pd

WINDOW_DAYS={
 "SCN_STRUCTURING":10,
 "SCN_DEPOSIT_SEND":3,
 "SCN_FAN_OUT":21,
 "SCN_FAN_IN":10,
 "SCN_CASH_WITHDRAWAL":7,
 "SCN_SMURFING":45,
 "SCN_UNUSUAL_AMOUNT_Z4":1,
 "SCN_SINGLE_LARGE_TRANSACTION":1,
 "SCN_GATHER_SCATTER":1,
 "SCN_HIGH_RISK_GEOGRAPHY":30,
 "SCN_SANCTIONED_GEOGRAPHY":30,
}
POLICY={"SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY"}

def aid(subject,scenario,seq):
 raw=f"V3|AGG|{subject}|{scenario}|{int(seq)}".encode()
 return "ALT-V3A-"+hashlib.sha1(raw).hexdigest()[:16].upper()

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--alerts",default="results/case_management_v3/alert.csv")
 p.add_argument("--bridge",default="results/case_management_v3/alert_transaction.csv")
 p.add_argument("--outdir",default="results/case_management_v3/aggregated")
 a=p.parse_args()

 alerts=pd.read_csv(a.alerts)
 bridge=pd.read_csv(a.bridge)
 alerts["alert_created_at"]=pd.to_datetime(alerts["alert_created_at"],errors="coerce")
 bridge["linked_at"]=pd.to_datetime(bridge["linked_at"],errors="coerce")
 if alerts["alert_created_at"].isna().any(): raise ValueError("Unparseable alert timestamps.")
 if set(alerts.scenario_id)-set(WINDOW_DAYS): raise ValueError("Missing scenario aggregation window.")

 # Preserve original transaction evidence before replacing alert headers.
 ev=bridge[["alert_id","transaction_id","scenario_id","linked_at"]].merge(
   alerts[["alert_id","primary_account_id","alert_amount","policy_flag","run_id"]],
   on="alert_id",how="inner",validate="many_to_one")

 # Scenario-specific inactivity sessionization.
 ev=ev.sort_values(["primary_account_id","scenario_id","linked_at","transaction_id"]).copy()
 prev=ev.groupby(["primary_account_id","scenario_id"])["linked_at"].shift()
 gap=(ev["linked_at"]-prev).dt.total_seconds()/86400
 lim=ev["scenario_id"].map(WINDOW_DAYS)
 new=prev.isna() | gap.gt(lim)
 ev["_seq"]=new.groupby([ev["primary_account_id"],ev["scenario_id"]]).cumsum().astype(int)
 ev["aggregated_alert_id"]=[aid(s,sc,q) for s,sc,q in zip(ev.primary_account_id,ev.scenario_id,ev._seq)]

 g=ev.groupby("aggregated_alert_id",sort=False)
 agg=g.agg(
   scenario_id=("scenario_id","first"),
   primary_account_id=("primary_account_id","first"),
   alert_created_at=("linked_at","min"),
   last_trigger_at=("linked_at","max"),
   transaction_count=("transaction_id","nunique"),
   trigger_event_count=("transaction_id","size"),
   alert_amount=("alert_amount","sum"),
   policy_flag=("policy_flag","max"),
   run_id=("run_id","first"),
 ).reset_index().rename(columns={"aggregated_alert_id":"alert_id"})
 agg["scenario_version"]="V3"
 agg["alert_status"]="NEW"
 agg["priority"]="MEDIUM"
 agg.loc[agg.policy_flag.astype(bool),"priority"]="HIGH"
 agg["aggregation_window_days"]=agg.scenario_id.map(WINDOW_DAYS)
 agg["trigger_reason"]=agg.apply(
   lambda r:(f"Frozen V3 {r.scenario_id}; {int(r.transaction_count)} unique triggering transaction(s); "
             f"{int(r.aggregation_window_days)}d inactivity-session aggregation"),axis=1)
 agg["assigned_to"]=pd.NA;agg["assigned_at"]=pd.NaT;agg["closed_at"]=pd.NaT
 agg["disposition_code"]=pd.NA;agg["case_id_current"]=pd.NA

 abr=ev[["aggregated_alert_id","transaction_id","scenario_id","linked_at"]].drop_duplicates(
      ["aggregated_alert_id","transaction_id"])
 abr=abr.rename(columns={"aggregated_alert_id":"alert_id"})
 abr["contribution_role"]="TRIGGER";abr["trigger_value"]=pd.NA

 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 agg.to_csv(out/"alert.csv",index=False)
 abr.to_csv(out/"alert_transaction.csv",index=False)

 summary=(agg.groupby("scenario_id",as_index=False)
          .agg(aggregated_alerts=("alert_id","count"),
               triggering_transactions=("transaction_count","sum"),
               unique_accounts=("primary_account_id","nunique"),
               median_transactions_per_alert=("transaction_count","median"),
               max_transactions_per_alert=("transaction_count","max")))
 summary["raw_trigger_alerts"]=summary.scenario_id.map(alerts.scenario_id.value_counts())
 summary["reduction_pct"]=1-summary.aggregated_alerts/summary.raw_trigger_alerts
 summary.to_csv(out/"aggregation_summary.csv",index=False)

 manifest={
  "portfolio_version":"V3","input_trigger_alerts":int(len(alerts)),
  "aggregated_alerts":int(len(agg)),
  "reduction_pct":float(1-len(agg)/len(alerts)),
  "unique_evidence_transactions":int(abr.transaction_id.nunique()),
  "scenario_windows_days":WINDOW_DAYS,
  "note":"Operational alert aggregation only; frozen V3 detection and HOLDOUT validation unchanged."
 }
 (out/"alert_aggregation_manifest.json").write_text(json.dumps(manifest,indent=2))

 print("=== V3 AGGREGATED ALERT LAYER ===")
 print(f"Input transaction-level scenario alerts: {len(alerts):,}")
 print(f"Aggregated investigation alerts: {len(agg):,}")
 print(f"Alert reduction: {1-len(agg)/len(alerts):.2%}")
 print(f"Unique evidence transactions retained: {abr.transaction_id.nunique():,}")
 print("\nBy scenario:")
 print(summary.to_string(index=False))
 print("\nOperational aggregation only. Frozen V3 detection/validation unchanged.")
 print("Saved:",out)

if __name__=="__main__": main()
