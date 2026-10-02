"""Build dashboard-ready DEVELOPMENT alert outputs from the frozen AML catalogue.

This first unified runner intentionally composes the already validated scenario
outputs instead of re-tuning thresholds. It reads DEVELOPMENT only and creates a
stable alert/KPI contract for the dashboard. HOLDOUT is never accessed.
"""
from pathlib import Path
import argparse
import pandas as pd
from scenario_config_frozen import FROZEN_SCENARIOS, assert_holdout_safe

FLAG_MAP = {
    "smurfing_cash_deposit": "SCN_SMURFING",
    "cash_withdrawal": "SCN_CASH_WITHDRAWAL",
    "fan_out": "SCN_FAN_OUT",
    "deposit_send": "SCN_DEPOSIT_SEND",
    "structuring": "SCN_STRUCTURING",
    "fan_in": "SCN_FAN_IN",
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("results/unified_dashboard"))
    a=p.parse_args()
    assert_holdout_safe()
    pending=[k for k,v in FROZEN_SCENARIOS.items() if v.get("status")!="frozen_development"]
    if pending: raise RuntimeError(f"Unfrozen scenarios: {pending}")

    print("Loading DEVELOPMENT for dashboard contract...")
    d=pd.read_csv(a.development)
    d["Transaction_timestamp"]=pd.to_datetime(
        pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")+" "+d["Time"].astype(str),
        errors="coerce")
    d=d.dropna(subset=["Transaction_timestamp"]).reset_index(drop=True)
    print(f"Development rows: {len(d):,} | AML: {int(d['Is_laundering'].sum()):,}")
    print("HOLDOUT is not accessed.")

    # Dashboard catalogue: one row per frozen scenario, using frozen DEVELOPMENT metrics.
    rows=[]
    for name,cfg in FROZEN_SCENARIOS.items():
        m=cfg["development_metrics"]
        recall=(m.get("target_recall",m.get("smurfing_recall",
                m.get("cash_withdrawal_recall",m.get("fan_out_recall",
                m.get("deposit_send_recall"))))))
        rows.append({
            "scenario":name,
            "status":cfg["status"],
            "triggered_transactions":m.get("triggered_transactions"),
            "trigger_rate":m.get("trigger_rate",m.get("channel_trigger_rate")),
            "precision":m.get("precision"),
            "target_recall":recall,
        })
    kpi=pd.DataFrame(rows).sort_values("scenario")

    # Base transaction table for the dashboard. Scenario flags are deliberately
    # nullable until the exact unified executor materializes each frozen rule.
    keep=["Transaction_timestamp","Sender_account","Receiver_account","Amount",
          "Payment_type","Is_laundering","Laundering_type"]
    keep=[c for c in keep if c in d.columns]
    tx=d[keep].copy()
    for flag in FLAG_MAP.values(): tx[flag]=pd.NA

    a.output_dir.mkdir(parents=True,exist_ok=True)
    kpi.to_csv(a.output_dir/"scenario_kpis_frozen_development.csv",index=False)
    tx.to_csv(a.output_dir/"transactions_dashboard_base.csv",index=False)

    print("\n=== DASHBOARD DATA CONTRACT ===")
    print(kpi.to_string(index=False))
    print(f"\nTransaction base rows: {len(tx):,}")
    print("Scenario flags are intentionally NULL in this staging file; no synthetic alerts were created.")
    print(f"Saved: {a.output_dir}")
    print("Next: materialize frozen scenario flags, then aggregate alert/account KPIs.")
    print("Development-only. HOLDOUT was not accessed.")

if __name__=="__main__": main()
