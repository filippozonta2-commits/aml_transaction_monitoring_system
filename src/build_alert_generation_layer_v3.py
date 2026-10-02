"""Build operational ALERT and ALERT_TRANSACTION tables from frozen V3 HOLDOUT outputs.

This is a downstream operationalization layer. It does not change scenario flags,
thresholds, portfolio membership, or the recorded HOLDOUT validation result.

Current alert grain is intentionally conservative:
    one triggered scenario x one triggering transaction = one alert header.
Window/entity aggregation can be introduced later as a separate case-management
design version without altering the validated V3 detection flags.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

CONTROLS=[
 "SCN_STRUCTURING","SCN_DEPOSIT_SEND","SCN_FAN_OUT","SCN_FAN_IN",
 "SCN_CASH_WITHDRAWAL","SCN_SMURFING","SCN_UNUSUAL_AMOUNT_Z4",
 "SCN_SINGLE_LARGE_TRANSACTION","SCN_GATHER_SCATTER",
 "SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY",
]
POLICY={"SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY"}

def spark_csv(path):
 fs=sorted(Path(path).glob("part-*.csv"))
 if not fs: raise FileNotFoundError(f"No Spark CSV part files under {path}")
 return pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)

def stable_alert_id(scenario,row_id):
 raw=f"V3|HOLDOUT|{scenario}|{int(row_id)}".encode()
 return "ALT-V3-"+hashlib.sha1(raw).hexdigest()[:16].upper()

def reason(sc,r):
 bits=[f"Frozen V3 control {sc} triggered"]
 if pd.notna(r.get("Amount")): bits.append(f"amount={float(r['Amount']):.2f}")
 if pd.notna(r.get("Payment_type")): bits.append(f"payment_type={r['Payment_type']}")
 if pd.notna(r.get("Sender_bank_location")) and pd.notna(r.get("Receiver_bank_location")):
  bits.append(f"route={r['Sender_bank_location']}->{r['Receiver_bank_location']}")
 return "; ".join(bits)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--holdout",default="data/temporal/SAML-D_holdout.csv")
 p.add_argument("--base",default="results/holdout_v3/non_deposit_send.csv")
 p.add_argument("--ds",default="results/holdout_v3/deposit_send")
 p.add_argument("--geo",default="results/holdout_v3/governed_geo")
 p.add_argument("--outdir",default="results/case_management_v3")
 a=p.parse_args()

 tx=pd.read_csv(a.holdout)
 tx["holdout_row_id"]=np.arange(len(tx),dtype="int64")
 flags=pd.read_csv(a.base).sort_values("holdout_row_id").reset_index(drop=True)
 ds=spark_csv(a.ds).sort_values("holdout_row_id").reset_index(drop=True)
 geo=spark_csv(a.geo).sort_values("holdout_row_id").reset_index(drop=True)
 for z in (flags,ds,geo):
  if not np.array_equal(flags.holdout_row_id.to_numpy(),z.holdout_row_id.to_numpy()):
   raise ValueError("HOLDOUT row ids do not align across V3 artifacts.")
 flags=flags.merge(ds,on="holdout_row_id",validate="one_to_one").merge(geo,on="holdout_row_id",validate="one_to_one")
 x=tx.merge(flags[["holdout_row_id"]+CONTROLS],on="holdout_row_id",validate="one_to_one")
 x["transaction_id"]=x["holdout_row_id"].map(lambda i:f"TXN-HO-{int(i):010d}")
 x["transaction_timestamp"]=pd.to_datetime(x["Date"].astype(str)+" "+x["Time"].astype(str),errors="coerce")

 long=x.melt(
  id_vars=["holdout_row_id","transaction_id","transaction_timestamp","Sender_account","Receiver_account","Amount",
           "Payment_currency","Received_currency","Sender_bank_location","Receiver_bank_location","Payment_type"],
  value_vars=CONTROLS,var_name="scenario_id",value_name="triggered_flag")
 trig=long[long.triggered_flag.eq(1)].copy()
 trig["alert_id"]=[stable_alert_id(s,i) for s,i in zip(trig.scenario_id,trig.holdout_row_id)]
 trig["scenario_version"]="V3"
 trig["alert_status"]="NEW"
 trig["priority"]=np.where(trig.scenario_id.isin(POLICY),"HIGH","MEDIUM")
 trig["policy_flag"]=trig.scenario_id.isin(POLICY)
 trig["primary_account_id"]=np.where(
   trig.scenario_id.isin(["SCN_STRUCTURING","SCN_FAN_IN"]),
   trig["Receiver_account"],trig["Sender_account"]).astype(str)
 trig["alert_created_at"]=trig["transaction_timestamp"]
 trig["trigger_reason"]=trig.apply(lambda r:reason(r.scenario_id,r),axis=1)
 trig["run_id"]="RUN-V3-HOLDOUT-VALIDATION"

 alerts=trig[["alert_id","scenario_id","scenario_version","alert_created_at","alert_status","priority",
             "primary_account_id","Amount","trigger_reason","policy_flag","run_id"]].rename(columns={"Amount":"alert_amount"})
 alerts["transaction_count"]=1
 alerts["assigned_to"]=pd.NA; alerts["assigned_at"]=pd.NaT; alerts["closed_at"]=pd.NaT
 alerts["disposition_code"]=pd.NA; alerts["case_id_current"]=pd.NA

 bridge=trig[["alert_id","transaction_id","scenario_id","transaction_timestamp"]].copy()
 bridge["contribution_role"]="TRIGGER";bridge["trigger_value"]=pd.NA
 bridge["linked_at"]=bridge["transaction_timestamp"]

 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 alerts.to_csv(out/"alert.csv",index=False)
 bridge.to_csv(out/"alert_transaction.csv",index=False)

 scenario_summary=(alerts.groupby("scenario_id",as_index=False)
                   .agg(alerts=("alert_id","count"),unique_primary_accounts=("primary_account_id","nunique"),
                        total_alert_amount=("alert_amount","sum"))
                   .sort_values("alerts",ascending=False))
 scenario_summary.to_csv(out/"alert_summary_by_scenario.csv",index=False)

 manifest={
  "portfolio_version":"V3","dataset":"HOLDOUT","alert_grain":"scenario_x_triggering_transaction",
  "alert_count":int(len(alerts)),"bridge_count":int(len(bridge)),
  "unique_triggering_transactions":int(bridge.transaction_id.nunique()),
  "controls":CONTROLS,
  "governance_note":"Downstream operationalization only; frozen V3 detection definitions are unchanged."
 }
 (out/"alert_generation_manifest.json").write_text(json.dumps(manifest,indent=2))

 print("=== V3 ALERT GENERATION LAYER ===")
 print(f"Alert headers: {len(alerts):,}")
 print(f"Alert-transaction rows: {len(bridge):,}")
 print(f"Unique triggering transactions: {bridge.transaction_id.nunique():,}")
 print("\nAlerts by scenario:")
 print(scenario_summary[["scenario_id","alerts","unique_primary_accounts"]].to_string(index=False))
 print("\nDetection flags/thresholds unchanged. This layer is operationalization only.")
 print("Saved:",out)

if __name__=="__main__": main()
