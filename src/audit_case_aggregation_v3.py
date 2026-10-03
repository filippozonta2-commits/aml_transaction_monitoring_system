"""Audit V3 case aggregation/sessionization behavior.

Downstream operational QA only. Does not alter frozen V3 detection or HOLDOUT validation.
Focus: long-lived / high-volume cases that may be produced by rolling 30-day sessionization.
"""
from pathlib import Path
import argparse, json
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cases",default="results/case_management_v3/case.csv")
    p.add_argument("--case-alert",default="results/case_management_v3/case_alert.csv")
    p.add_argument("--alerts",default="results/case_management_v3/alert.csv")
    p.add_argument("--outdir",default="results/case_management_v3/case_audit")
    a=p.parse_args()

    cases=pd.read_csv(a.cases)
    ca=pd.read_csv(a.case_alert)
    alerts=pd.read_csv(a.alerts)

    for c in ["case_created_at","last_alert_at"]:
        cases[c]=pd.to_datetime(cases[c],errors="coerce")
    alerts["alert_created_at"]=pd.to_datetime(alerts["alert_created_at"],errors="coerce")

    cases["case_duration_days"]=(cases["last_alert_at"]-cases["case_created_at"]).dt.total_seconds()/86400
    cases["high_volume_50_plus"]=cases["alert_count"].ge(50)
    cases["high_volume_100_plus"]=cases["alert_count"].ge(100)
    cases["long_lived_90_plus"]=cases["case_duration_days"].gt(90)
    cases["long_lived_180_plus"]=cases["case_duration_days"].gt(180)

    # Alert-level composition per case.
    comp=(ca[["case_id","alert_id"]]
          .merge(alerts[["alert_id","scenario_id","policy_flag","alert_created_at"]],
                 on="alert_id",how="left",validate="many_to_one"))
    scen=(comp.groupby(["case_id","scenario_id"]).size()
          .rename("scenario_alerts").reset_index())
    dominant=(scen.sort_values(["case_id","scenario_alerts"],ascending=[True,False])
              .drop_duplicates("case_id")
              .rename(columns={"scenario_id":"dominant_scenario",
                               "scenario_alerts":"dominant_scenario_alerts"})
              [["case_id","dominant_scenario","dominant_scenario_alerts"]])
    audit=cases.merge(dominant,on="case_id",how="left")
    audit["dominant_share"]=audit["dominant_scenario_alerts"]/audit["alert_count"]

    # Calculate largest gap inside each case: useful to validate sessionization.
    comp=comp.sort_values(["case_id","alert_created_at","alert_id"])
    comp["_prev"]=comp.groupby("case_id")["alert_created_at"].shift()
    comp["_gap_days"]=(comp["alert_created_at"]-comp["_prev"]).dt.total_seconds()/86400
    gaps=comp.groupby("case_id")["_gap_days"].max().rename("max_internal_gap_days")
    audit=audit.join(gaps,on="case_id")

    top=audit.sort_values(["alert_count","case_duration_days"],ascending=False).head(100)
    flagged=audit[
        audit["high_volume_50_plus"] | audit["long_lived_90_plus"]
    ].sort_values(["alert_count","case_duration_days"],ascending=False)

    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    audit.to_csv(out/"case_aggregation_audit.csv",index=False)
    top.to_csv(out/"top_100_cases.csv",index=False)
    flagged.to_csv(out/"flagged_cases.csv",index=False)
    scen.to_csv(out/"case_scenario_composition.csv",index=False)

    summary={
        "cases":int(len(audit)),
        "duration_median_days":float(audit.case_duration_days.median()),
        "duration_p95_days":float(audit.case_duration_days.quantile(.95)),
        "duration_p99_days":float(audit.case_duration_days.quantile(.99)),
        "duration_max_days":float(audit.case_duration_days.max()),
        "cases_50_plus_alerts":int(audit.high_volume_50_plus.sum()),
        "cases_100_plus_alerts":int(audit.high_volume_100_plus.sum()),
        "cases_over_90_days":int(audit.long_lived_90_plus.sum()),
        "cases_over_180_days":int(audit.long_lived_180_plus.sum()),
        "max_internal_gap_days":float(audit.max_internal_gap_days.max()),
        "note":"Audit only. Do not change frozen V3 detection based on case-management behavior."
    }
    (out/"case_aggregation_audit_manifest.json").write_text(json.dumps(summary,indent=2))

    print("=== V3 CASE AGGREGATION AUDIT ===")
    print(f"Cases: {len(audit):,}")
    print("\nCase duration (days):")
    print(audit["case_duration_days"].describe(percentiles=[.5,.75,.9,.95,.99]).to_string())
    print(f"\nCases >=50 alerts: {audit.high_volume_50_plus.sum():,}")
    print(f"Cases >=100 alerts: {audit.high_volume_100_plus.sum():,}")
    print(f"Cases >90 days: {audit.long_lived_90_plus.sum():,}")
    print(f"Cases >180 days: {audit.long_lived_180_plus.sum():,}")
    print(f"Max internal alert gap: {audit.max_internal_gap_days.max():.2f} days")
    print("\nTop 15 cases:")
    cols=["case_id","subject_id","alert_count","transaction_count","scenario_count",
          "case_duration_days","dominant_scenario","dominant_share","policy_flag"]
    print(top[cols].head(15).to_string(index=False))
    print("\nAudit only. Frozen V3 detection/validation unchanged.")
    print("Saved:",out)

if __name__=="__main__":
    main()
