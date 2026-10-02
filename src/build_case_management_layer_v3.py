"""Build CASE and CASE_ALERT operational tables from V3 alert outputs.

Operational design only. Frozen V3 detection flags and HOLDOUT validation are untouched.

Case grouping rule (V1):
- subject = ALERT.primary_account_id
- sort alerts chronologically per subject
- start a new case when the gap from the PREVIOUS alert is > 30 days
This is sessionization, not a fixed 30-day bucket, so sustained activity remains in one case.
"""
from pathlib import Path
import argparse, hashlib, json
import pandas as pd

WINDOW_DAYS=30

def stable_case_id(subject, seq):
    raw=f"V3|CASE|{subject}|{int(seq)}".encode()
    return "CAS-V3-"+hashlib.sha1(raw).hexdigest()[:16].upper()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--alerts",default="results/case_management_v3/alert.csv")
    p.add_argument("--bridge",default="results/case_management_v3/alert_transaction.csv")
    p.add_argument("--outdir",default="results/case_management_v3")
    a=p.parse_args()

    alerts=pd.read_csv(a.alerts)
    bridge=pd.read_csv(a.bridge)
    alerts["alert_created_at"]=pd.to_datetime(alerts["alert_created_at"],errors="coerce")
    if alerts["alert_created_at"].isna().any():
        raise ValueError("Some alert_created_at values could not be parsed.")
    if alerts["alert_id"].duplicated().any():
        raise ValueError("alert_id must be unique before case grouping.")
    if set(bridge["alert_id"])-set(alerts["alert_id"]):
        raise ValueError("ALERT_TRANSACTION contains alert_ids absent from ALERT.")

    x=alerts.sort_values(["primary_account_id","alert_created_at","alert_id"]).copy()
    prev=x.groupby("primary_account_id")["alert_created_at"].shift()
    new_case=prev.isna() | ((x["alert_created_at"]-prev).dt.total_seconds() > WINDOW_DAYS*86400)
    x["_case_seq"]=new_case.groupby(x["primary_account_id"]).cumsum().astype(int)
    x["case_id"]=[stable_case_id(s,q) for s,q in zip(x["primary_account_id"],x["_case_seq"])]

    case_alert=x[["case_id","alert_id"]].copy()
    case_alert["link_reason"]="SAME_ACCOUNT_30D_SESSION"
    case_alert["linked_by"]="SYSTEM"
    case_alert["linked_at"]=x["alert_created_at"].to_numpy()
    case_alert["active_flag"]=True

    tx_per_alert=bridge.groupby("alert_id")["transaction_id"].nunique().rename("_tx_n")
    x=x.join(tx_per_alert,on="alert_id")
    x["_tx_n"]=x["_tx_n"].fillna(0).astype(int)

    grouped=x.groupby("case_id",sort=False)
    cases=grouped.agg(
        subject_id=("primary_account_id","first"),
        case_created_at=("alert_created_at","min"),
        last_alert_at=("alert_created_at","max"),
        alert_count=("alert_id","count"),
        scenario_count=("scenario_id","nunique"),
        transaction_links=(" _tx_n" if False else "_tx_n","sum"),
        total_alert_amount=("alert_amount","sum"),
        policy_flag=("policy_flag","max"),
    ).reset_index()

    scenario_lists=(grouped["scenario_id"].agg(lambda s:"|".join(sorted(set(s)))).rename("scenarios"))
    cases=cases.join(scenario_lists,on="case_id")
    cases["subject_type"]="ACCOUNT"
    cases["case_status"]="OPEN"
    cases["case_priority"]="MEDIUM"
    cases.loc[cases["policy_flag"].astype(bool),"case_priority"]="HIGH"
    cases.loc[cases["scenario_count"].ge(3),"case_priority"]="HIGH"
    cases["case_summary"]=cases.apply(
        lambda r:f"{int(r.alert_count)} alert(s), {int(r.scenario_count)} scenario(s): {r.scenarios}",axis=1)
    cases["assigned_to"]=pd.NA
    cases["assigned_at"]=pd.NaT
    cases["investigation_started_at"]=pd.NaT
    cases["decision_at"]=pd.NaT
    cases["closed_at"]=pd.NaT
    cases["disposition_code"]=pd.NA
    cases["escalation_flag"]=False
    cases["sar_consideration_flag"]=False
    cases["investigator_notes"]=pd.NA
    cases["created_by"]="SYSTEM"
    cases["last_updated_at"]=cases["last_alert_at"]

    # Exact unique transaction count per case, avoiding double-counting the same transaction
    # when it triggered multiple alerts/scenarios.
    cb=case_alert[["case_id","alert_id"]].merge(
        bridge[["alert_id","transaction_id"]],on="alert_id",how="left",validate="one_to_many")
    unique_tx=cb.groupby("case_id")["transaction_id"].nunique().rename("transaction_count")
    cases=cases.drop(columns=["transaction_links"]).join(unique_tx,on="case_id")
    cases["transaction_count"]=cases["transaction_count"].fillna(0).astype(int)

    # Convenience pointer in ALERT; CASE_ALERT remains canonical relationship.
    alert_case=case_alert.set_index("alert_id")["case_id"]
    alerts["case_id_current"]=alerts["alert_id"].map(alert_case)

    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    cases.to_csv(out/"case.csv",index=False)
    case_alert.to_csv(out/"case_alert.csv",index=False)
    alerts.to_csv(out/"alert.csv",index=False)

    summary={
        "portfolio_version":"V3",
        "grouping_rule":f"same primary account; new case after >{WINDOW_DAYS}-day gap from previous alert",
        "alerts":int(len(alerts)),
        "cases":int(len(cases)),
        "compression_ratio_alerts_per_case":float(len(alerts)/len(cases)),
        "multi_alert_cases":int(cases["alert_count"].gt(1).sum()),
        "multi_scenario_cases":int(cases["scenario_count"].gt(1).sum()),
        "policy_cases":int(cases["policy_flag"].astype(bool).sum()),
        "governance_note":"Case grouping is downstream operational logic; it does not alter frozen V3 detection or HOLDOUT metrics."
    }
    (out/"case_generation_manifest.json").write_text(json.dumps(summary,indent=2))

    print("=== V3 CASE MANAGEMENT LAYER ===")
    print(f"Alerts linked: {len(alerts):,}")
    print(f"Cases created: {len(cases):,}")
    print(f"Compression: {len(alerts)/len(cases):.2f} alerts/case")
    print(f"Multi-alert cases: {cases['alert_count'].gt(1).sum():,} ({cases['alert_count'].gt(1).mean():.2%})")
    print(f"Multi-scenario cases: {cases['scenario_count'].gt(1).sum():,} ({cases['scenario_count'].gt(1).mean():.2%})")
    print(f"Policy-linked cases: {cases['policy_flag'].astype(bool).sum():,}")
    print("\nCase alert-count distribution:")
    print(cases["alert_count"].describe(percentiles=[.5,.75,.9,.95,.99]).to_string())
    print("\nOperational grouping only. Frozen V3 detection/validation unchanged.")
    print("Saved:",out)

if __name__=="__main__":
    main()
