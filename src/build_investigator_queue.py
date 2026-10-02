"""Build the frozen 72h inactivity-based DEVELOPMENT investigator queue.

Operational fields are separated from evaluation-only AML labels so the latter
cannot leak into future prioritization/ML features. HOLDOUT is not accessed.
"""
from pathlib import Path
import argparse
import pandas as pd

GAP_HOURS=72
ROLE_MAP={"SCN_SMURFING":"Sender_account","SCN_CASH_WITHDRAWAL":"Sender_account",
          "SCN_FAN_OUT":"Sender_account","SCN_STRUCTURING":"Sender_account",
          "SCN_FAN_IN":"Receiver_account","SCN_DEPOSIT_SEND":"Receiver_account"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--alerts",type=Path,default=Path("results/unified_dashboard/unified_alert_table.csv"))
    p.add_argument("--out",type=Path,default=Path("results/investigator_queue"))
    a=p.parse_args()
    print("Loading unified DEVELOPMENT alerts...")
    d=pd.read_csv(a.alerts)
    x=d[d["ANY_SCENARIO_ALERT"].eq(1)].copy()
    x["ts"]=pd.to_datetime(x["ts"],errors="coerce")
    if x["ts"].isna().any(): raise RuntimeError("NULL timestamps found in alerted transactions.")

    parts=[]
    for scn,acct in ROLE_MAP.items():
        z=x[x[scn].eq(1)][["development_row_id","ts","Amount","Is_laundering","Laundering_type",acct]].copy()
        z=z.rename(columns={acct:"account_id"})
        z["scenario"]=scn
        parts.append(z)
    e=pd.concat(parts,ignore_index=True).dropna(subset=["account_id"])
    e=e.sort_values(["account_id","ts","development_row_id"]).reset_index(drop=True)

    prev=e.groupby("account_id")["ts"].shift()
    e["new_case"]=(prev.isna()|((e["ts"]-prev).dt.total_seconds()/3600>GAP_HOURS)).astype("int8")
    e["case_seq"]=e["new_case"].groupby(e["account_id"]).cumsum().astype(int)
    e["CASE_ID"]="CASE-"+e["account_id"].astype(str)+"-"+e["case_seq"].astype(str).str.zfill(3)

    # Operational queue: no AML truth columns.
    q=e.groupby(["CASE_ID","account_id"]).agg(
        case_start=("ts","min"),case_end=("ts","max"),
        scenario_events=("development_row_id","size"),
        transaction_alerts=("development_row_id","nunique"),
        total_alert_amount=("Amount","sum"),
        active_scenarios=("scenario","nunique"),
    ).reset_index()
    q["duration_hours"]=(q["case_end"]-q["case_start"]).dt.total_seconds()/3600
    scenarios=(e.groupby("CASE_ID")["scenario"]
               .agg(lambda s:" | ".join(sorted(set(s)))).rename("scenario_list"))
    q=q.merge(scenarios,on="CASE_ID",how="left")
    q["multi_scenario_flag"]=(q["active_scenarios"]>1).astype("int8")
    q["case_status"]="OPEN"
    q["case_policy"]="72h_inactivity"

    # Evaluation table kept separate to prevent label leakage.
    ev=e.groupby("CASE_ID").agg(
        aml_positive=("Is_laundering",lambda s:int((s==1).any())),
        aml_event_count=("Is_laundering","sum"),
    ).reset_index()
    types=(e.loc[e["Is_laundering"].eq(1)].groupby("CASE_ID")["Laundering_type"]
           .agg(lambda s:" | ".join(sorted(set(s.dropna().astype(str))))).rename("aml_typologies"))
    ev=ev.merge(types,on="CASE_ID",how="left")

    # Link table preserves lineage from cases back to transaction/scenario events.
    links=e[["CASE_ID","account_id","development_row_id","ts","scenario"]].copy()

    a.out.mkdir(parents=True,exist_ok=True)
    q.to_csv(a.out/"investigator_queue.csv",index=False)
    ev.to_csv(a.out/"case_evaluation_labels.csv",index=False)
    links.to_csv(a.out/"case_transaction_lineage.csv",index=False)

    print("\n=== INVESTIGATOR QUEUE — 72H INACTIVITY POLICY ===")
    print(f"Cases: {len(q):,}")
    print(f"Accounts: {q.account_id.nunique():,}")
    print(f"Multi-scenario cases: {q.multi_scenario_flag.sum():,}")
    print(f"Median transaction alerts/case: {q.transaction_alerts.median():.1f}")
    print(f"P90 transaction alerts/case: {q.transaction_alerts.quantile(.90):.1f}")
    print(f"AML-positive cases (evaluation only): {ev.aml_positive.sum():,}")
    print(f"Operational queue columns: {list(q.columns)}")
    print("\nAML truth is stored separately in case_evaluation_labels.csv to avoid feature leakage.")
    print(f"Saved: {a.out}")
    print("Frozen policy: new case after >72h inactivity. Development-only. HOLDOUT was not accessed.")

if __name__=="__main__": main()
