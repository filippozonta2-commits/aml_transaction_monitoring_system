"""Build independent ML V1 investigation cases from frozen aggregated ML alerts.

Operational case management only: never reads AML labels or changes detection.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd

def case_id(subject, sequence):
    token = f"MLV1|CASE|{subject}|{int(sequence)}".encode()
    return "CAS-ML1-" + hashlib.sha1(token).hexdigest()[:16].upper()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alerts", type=Path, default=Path("results/ml_case_management_v1/aggregated/alert.csv"))
    p.add_argument("--bridge", type=Path, default=Path("results/ml_case_management_v1/aggregated/alert_transaction.csv"))
    p.add_argument("--outdir", type=Path, default=Path("results/ml_case_management_v1/cases"))
    p.add_argument("--window-days", type=int, default=30)
    a = p.parse_args()
    if a.window_days < 1:
        raise ValueError("window-days must be positive")
    alerts = pd.read_csv(a.alerts)
    bridge = pd.read_csv(a.bridge)
    if alerts.empty:
        raise ValueError("No frozen ML alerts available to create cases")
    if alerts.alert_id.duplicated().any():
        raise ValueError("Duplicate ML alert IDs")
    if bridge.empty or set(bridge.alert_id) - set(alerts.alert_id):
        raise ValueError("Missing or orphaned ML alert-to-transaction evidence")
    alerts["alert_created_at"] = pd.to_datetime(alerts.alert_created_at, errors="raise")
    alerts["last_trigger_at"] = pd.to_datetime(alerts.last_trigger_at, errors="raise")
    z = alerts.sort_values(["primary_account_id", "alert_created_at", "alert_id"]).copy()
    previous = z.groupby("primary_account_id").alert_created_at.shift()
    gap = (z.alert_created_at - previous).dt.total_seconds() / 86400
    new_case = previous.isna() | gap.gt(a.window_days)
    z["_sequence"] = new_case.groupby(z.primary_account_id).cumsum().astype(int)
    z["case_id"] = [case_id(s, n) for s, n in zip(z.primary_account_id, z._sequence)]
    case_alert = z[["case_id", "alert_id"]].copy()
    case_alert["link_reason"] = f"SAME_ACCOUNT_{a.window_days}D_SESSION"
    case_alert["linked_at"] = z.alert_created_at.to_numpy()
    case_alert["linked_by"] = "SYSTEM"
    tx_links = case_alert.merge(bridge[["alert_id", "transaction_id"]].drop_duplicates(), on="alert_id", validate="one_to_many")
    unique_tx = tx_links.groupby("case_id").transaction_id.nunique()
    cases = z.groupby("case_id", sort=False).agg(
        subject_id=("primary_account_id", "first"),
        case_created_at=("alert_created_at", "min"),
        last_alert_at=("last_trigger_at", "max"),
        alert_count=("alert_id", "nunique"),
        total_alert_amount=("alert_amount", "sum"),
        ml_probability_max=("ml_probability_max", "max"),
        ml_probability_mean=("ml_probability_mean", "mean"),
    ).reset_index()
    cases["transaction_count"] = cases.case_id.map(unique_tx).fillna(0).astype(int)
    cases["case_status"] = "OPEN"
    cases["detection_engine"] = "ML_V1"
    cases["case_priority"] = "UNASSIGNED"
    cases["case_summary"] = cases.apply(lambda x: f"{x.alert_count} ML alert(s); {x.transaction_count} unique transaction(s); maximum probability {x.ml_probability_max:.4f}", axis=1)
    cases = cases.sort_values(["ml_probability_max", "alert_count", "last_alert_at", "case_id"], ascending=[False, False, False, True]).reset_index(drop=True)
    cases["queue_rank"] = cases.index + 1
    cases["priority_reason"] = "Ranked by maximum frozen ML probability; operational ranking, not a calibrated case risk score"
    if cases.case_id.duplicated().any() or len(case_alert) != len(alerts):
        raise ValueError("Case lineage validation failed")
    a.outdir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(a.outdir / "case.csv", index=False)
    cases.to_csv(a.outdir / "case_priority_queue.csv", index=False)
    case_alert.to_csv(a.outdir / "case_alert.csv", index=False)
    manifest = {
        "detector": "Frozen ML Detection V1",
        "aggregation": "Account-centric inactivity sessions; separate from rule-based cases",
        "case_window_days": a.window_days,
        "aggregated_alerts": int(len(alerts)),
        "cases": int(len(cases)),
        "unique_evidence_transactions": int(tx_links.transaction_id.nunique()),
        "aml_labels_used": False,
        "model_or_threshold_changed": False,
    }
    (a.outdir / "case_generation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"ML aggregated alerts: {len(alerts):,} | ML cases: {len(cases):,} | Evidence transactions: {manifest['unique_evidence_transactions']:,}")
    print(f"Saved to {a.outdir}")

if __name__ == "__main__":
    main()
