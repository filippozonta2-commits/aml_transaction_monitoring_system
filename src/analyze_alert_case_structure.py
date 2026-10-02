"""Diagnose account/time structure of unified DEVELOPMENT alerts before case formation.

No case window is selected here. The purpose is to measure alert concentration,
scenario combinations and temporal spacing so case aggregation can be justified.
HOLDOUT is not accessed.
"""
from pathlib import Path
import argparse
import pandas as pd

FLAGS=["SCN_SMURFING","SCN_CASH_WITHDRAWAL","SCN_FAN_OUT","SCN_STRUCTURING","SCN_FAN_IN","SCN_DEPOSIT_SEND"]

def parse_ts(df):
    date=pd.to_datetime(df["Date"],errors="coerce")
    return pd.to_datetime(date.dt.strftime("%Y-%m-%d")+" "+df["Time"].astype(str).str.strip(),errors="coerce")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--alerts",type=Path,default=Path("results/unified_dashboard/unified_alert_table.csv"))
    p.add_argument("--out",type=Path,default=Path("results/case_structure"))
    a=p.parse_args()

    print("Loading unified DEVELOPMENT alert table...")
    d=pd.read_csv(a.alerts)
    x=d[d["ANY_SCENARIO_ALERT"].eq(1)].copy()
    x["ts"]=parse_ts(x)
    if x["ts"].isna().any():
        raise RuntimeError(f"NULL timestamps among alerts: {x['ts'].isna().sum():,}")

    # Anchor each transaction alert to the account whose behavior is being investigated.
    # Receiver-based scenarios: Fan-In and Deposit-Send. Sender-oriented/default:
    # Smurfing, Cash Withdrawal, Fan-Out, Structuring.
    # Multi-scenario rows can implicate both roles, so create account-scenario events
    # rather than forcing one arbitrary account onto the transaction.
    events=[]
    role_map={
        "SCN_SMURFING":"Sender_account",
        "SCN_CASH_WITHDRAWAL":"Sender_account",
        "SCN_FAN_OUT":"Sender_account",
        "SCN_STRUCTURING":"Sender_account",
        "SCN_FAN_IN":"Receiver_account",
        "SCN_DEPOSIT_SEND":"Receiver_account",
    }
    for f,acct_col in role_map.items():
        z=x[x[f].eq(1)][["development_row_id","ts","Is_laundering","Laundering_type",acct_col]].copy()
        z=z.rename(columns={acct_col:"account_id"})
        z["scenario"]=f
        events.append(z)
    e=pd.concat(events,ignore_index=True)
    e=e.dropna(subset=["account_id"]).sort_values(["account_id","ts","development_row_id"])

    acct=e.groupby("account_id").agg(
        scenario_events=("development_row_id","size"),
        unique_alert_transactions=("development_row_id","nunique"),
        active_scenarios=("scenario","nunique"),
        first_alert=("ts","min"),
        last_alert=("ts","max"),
        aml_events=("Is_laundering","sum"),
    ).reset_index()
    acct["active_days"]=(acct["last_alert"]-acct["first_alert"]).dt.total_seconds()/86400

    combos=(e.groupby("account_id")["scenario"].agg(lambda s:" + ".join(sorted(set(s))))
            .value_counts().rename_axis("scenario_combination").reset_index(name="accounts"))
    combos["share_of_accounts"]=combos["accounts"]/acct.shape[0]

    # Consecutive event gaps: empirical input for a later case-window decision.
    e["previous_ts"]=e.groupby("account_id")["ts"].shift()
    e["gap_hours"]=(e["ts"]-e["previous_ts"]).dt.total_seconds()/3600
    gaps=e["gap_hours"].dropna()
    gap_summary=pd.DataFrame([{
        "events_with_prior_alert":len(gaps),
        "median_gap_hours":gaps.median() if len(gaps) else None,
        "p75_gap_hours":gaps.quantile(.75) if len(gaps) else None,
        "p90_gap_hours":gaps.quantile(.90) if len(gaps) else None,
        "p95_gap_hours":gaps.quantile(.95) if len(gaps) else None,
        "p99_gap_hours":gaps.quantile(.99) if len(gaps) else None,
    }])

    concentration=pd.DataFrame({
        "metric":["accounts","median_events","p75_events","p90_events","p95_events","p99_events","max_events"],
        "value":[len(acct),acct.scenario_events.median(),acct.scenario_events.quantile(.75),
                 acct.scenario_events.quantile(.90),acct.scenario_events.quantile(.95),
                 acct.scenario_events.quantile(.99),acct.scenario_events.max()]
    })

    a.out.mkdir(parents=True,exist_ok=True)
    acct.to_csv(a.out/"account_alert_structure.csv",index=False)
    combos.to_csv(a.out/"account_scenario_combinations.csv",index=False)
    gap_summary.to_csv(a.out/"consecutive_alert_gap_summary.csv",index=False)
    concentration.to_csv(a.out/"account_alert_concentration.csv",index=False)

    print("\n=== ACCOUNT ALERT CONCENTRATION ===")
    print(concentration.to_string(index=False))
    print("\n=== CONSECUTIVE ALERT GAPS ===")
    print(gap_summary.to_string(index=False))
    print("\n=== TOP ACCOUNT SCENARIO COMBINATIONS ===")
    print(combos.head(15).to_string(index=False))
    print(f"\nAccount-scenario events: {len(e):,}")
    print(f"Unique investigated accounts: {len(acct):,}")
    print(f"Accounts with >1 active scenario: {(acct.active_scenarios.gt(1)).sum():,}")
    print(f"Saved: {a.out}")
    print("No case window was selected. Development-only. HOLDOUT was not accessed.")

if __name__=="__main__":
    main()
