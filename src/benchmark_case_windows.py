"""Benchmark inactivity-based case formation windows on DEVELOPMENT alerts.

Compares 24h, 72h and 168h inactivity gaps without selecting a final policy.
A new case begins when an investigated account's next scenario-event occurs
more than the candidate gap after its prior event. HOLDOUT is not accessed.
"""
from pathlib import Path
import argparse
import pandas as pd

FLAGS=["SCN_SMURFING","SCN_CASH_WITHDRAWAL","SCN_FAN_OUT","SCN_STRUCTURING","SCN_FAN_IN","SCN_DEPOSIT_SEND"]
ROLE_MAP={"SCN_SMURFING":"Sender_account","SCN_CASH_WITHDRAWAL":"Sender_account",
          "SCN_FAN_OUT":"Sender_account","SCN_STRUCTURING":"Sender_account",
          "SCN_FAN_IN":"Receiver_account","SCN_DEPOSIT_SEND":"Receiver_account"}

def events_from_alerts(d):
    x=d[d["ANY_SCENARIO_ALERT"].eq(1)].copy()
    x["ts"]=pd.to_datetime(x["ts"],errors="coerce")
    if x["ts"].isna().any(): raise RuntimeError(f"NULL alert timestamps: {x['ts'].isna().sum():,}")
    parts=[]
    for f,acct in ROLE_MAP.items():
        z=x[x[f].eq(1)][["development_row_id","ts","Is_laundering","Laundering_type",acct]].copy()
        z=z.rename(columns={acct:"account_id"}); z["scenario"]=f; parts.append(z)
    return pd.concat(parts,ignore_index=True).dropna(subset=["account_id"]).sort_values(["account_id","ts","development_row_id"])

def form_cases(e,gap_h):
    x=e.copy()
    prev=x.groupby("account_id")["ts"].shift()
    new=(prev.isna()|((x["ts"]-prev).dt.total_seconds()/3600>gap_h)).astype(int)
    x["case_seq"]=new.groupby(x["account_id"]).cumsum()
    x["case_id"]=x["account_id"].astype(str)+"_"+x["case_seq"].astype(str)
    c=x.groupby(["account_id","case_id"]).agg(
        first_alert=("ts","min"),last_alert=("ts","max"),
        scenario_events=("development_row_id","size"),
        unique_alert_transactions=("development_row_id","nunique"),
        active_scenarios=("scenario","nunique"),
        aml_events=("Is_laundering","sum"),
    ).reset_index()
    c["duration_hours"]=(c["last_alert"]-c["first_alert"]).dt.total_seconds()/3600
    return x,c

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--alerts",type=Path,default=Path("results/unified_dashboard/unified_alert_table.csv"))
    p.add_argument("--out",type=Path,default=Path("results/case_window_benchmark"))
    a=p.parse_args()
    d=pd.read_csv(a.alerts); e=events_from_alerts(d)
    aml_tx=set(d.loc[d["Is_laundering"].eq(1)&d["ANY_SCENARIO_ALERT"].eq(1),"development_row_id"])
    rows=[]
    a.out.mkdir(parents=True,exist_ok=True)
    for h in [24,72,168]:
        ev,c=form_cases(e,h)
        accounts_multi=int(c.groupby("account_id").size().gt(1).sum())
        aml_cases=int(c["aml_events"].gt(0).sum())
        aml_tx_in_cases=set(ev.loc[ev["Is_laundering"].eq(1),"development_row_id"])
        rows.append({
            "inactivity_gap_hours":h,"cases":len(c),"accounts":c.account_id.nunique(),
            "cases_per_account":len(c)/c.account_id.nunique(),
            "median_events_per_case":c.scenario_events.median(),
            "p90_events_per_case":c.scenario_events.quantile(.90),
            "median_case_duration_hours":c.duration_hours.median(),
            "p90_case_duration_hours":c.duration_hours.quantile(.90),
            "accounts_with_multiple_cases":accounts_multi,
            "multi_scenario_cases":int(c.active_scenarios.gt(1).sum()),
            "aml_positive_cases":aml_cases,
            "alerted_aml_tx_coverage":len(aml_tx_in_cases & aml_tx)/len(aml_tx) if aml_tx else 0,
        })
        c.to_csv(a.out/f"cases_gap_{h}h.csv",index=False)
    s=pd.DataFrame(rows)
    s.to_csv(a.out/"case_window_benchmark.csv",index=False)
    print("\n=== CASE WINDOW BENCHMARK ===")
    print(s.to_string(index=False))
    print(f"\nInput account-scenario events: {len(e):,}")
    print(f"Unique investigated accounts: {e.account_id.nunique():,}")
    print("No final case window selected. Development-only. HOLDOUT was not accessed.")
    print(f"Saved: {a.out}")

if __name__=="__main__": main()
