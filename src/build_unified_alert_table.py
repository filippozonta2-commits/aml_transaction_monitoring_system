"""Build the six-scenario unified DEVELOPMENT alert table.

Merges the reconciled non-Deposit-Send materialization with the validated
PySpark Deposit-Send flags. No thresholds are tuned and HOLDOUT is not read.
"""
from pathlib import Path
import argparse
import glob
import pandas as pd

FLAGS=[
    "SCN_SMURFING","SCN_CASH_WITHDRAWAL","SCN_FAN_OUT",
    "SCN_STRUCTURING","SCN_FAN_IN","SCN_DEPOSIT_SEND",
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=Path,default=Path("results/unified_dashboard/materialized_non_deposit_send.csv"))
    p.add_argument("--deposit-send-dir",type=Path,default=Path("results/unified_dashboard/deposit_send_flags"))
    p.add_argument("--output-dir",type=Path,default=Path("results/unified_dashboard"))
    a=p.parse_args()

    print("Loading reconciled frozen scenario outputs...")
    base=pd.read_csv(a.base)
    parts=sorted(glob.glob(str(a.deposit_send_dir/"part-*.csv")))
    if not parts:
        raise FileNotFoundError(f"No Spark part CSV found in {a.deposit_send_dir}")
    ds=pd.concat([pd.read_csv(x) for x in parts],ignore_index=True)

    if base["development_row_id"].duplicated().any():
        raise RuntimeError("Duplicate development_row_id in base materialization.")
    if ds["development_row_id"].duplicated().any():
        raise RuntimeError("Duplicate development_row_id in Deposit-Send materialization.")
    if len(base)!=len(ds):
        raise RuntimeError(f"Row-count mismatch: base={len(base):,}, deposit_send={len(ds):,}")

    # Recompute aggregate flags after adding the sixth scenario.
    base=base.drop(columns=["SCENARIO_COUNT","ANY_SCENARIO_ALERT"],errors="ignore")
    out=base.merge(ds[["development_row_id","SCN_DEPOSIT_SEND"]],
                   on="development_row_id",how="left",validate="one_to_one")
    if out["SCN_DEPOSIT_SEND"].isna().any():
        raise RuntimeError("Deposit-Send merge left unmatched development rows.")
    out["SCN_DEPOSIT_SEND"]=out["SCN_DEPOSIT_SEND"].astype("int8")
    out["SCENARIO_COUNT"]=out[FLAGS].sum(axis=1).astype("int8")
    out["ANY_SCENARIO_ALERT"]=(out["SCENARIO_COUNT"]>0).astype("int8")

    n=len(out); alerted=out["ANY_SCENARIO_ALERT"].eq(1)
    aml=out["Is_laundering"].eq(1)
    summary=[]
    for f in FLAGS:
        m=out[f].eq(1)
        summary.append({
            "Scenario":f,
            "Triggered_transactions":int(m.sum()),
            "Trigger_rate":float(m.mean()),
            "AML_cases":int((m&aml).sum()),
            "Precision":float((m&aml).sum()/m.sum()) if m.sum() else 0.0,
        })
    summary=pd.DataFrame(summary)

    dist=(out.loc[alerted,"SCENARIO_COUNT"].value_counts().sort_index()
          .rename_axis("Scenario_count").reset_index(name="Alerted_transactions"))
    dist["Share_of_alerted"]=dist["Alerted_transactions"]/alerted.sum()

    # Pairwise overlap is useful for dashboard governance and deduplication.
    overlap=[]
    for i,a1 in enumerate(FLAGS):
        for a2 in FLAGS[i+1:]:
            both=out[a1].eq(1)&out[a2].eq(1)
            overlap.append({"Scenario_A":a1,"Scenario_B":a2,"Overlap_transactions":int(both.sum())})
    overlap=pd.DataFrame(overlap).sort_values("Overlap_transactions",ascending=False)

    a.output_dir.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output_dir/"unified_alert_table.csv",index=False)
    summary.to_csv(a.output_dir/"unified_scenario_summary.csv",index=False)
    dist.to_csv(a.output_dir/"scenario_count_distribution.csv",index=False)
    overlap.to_csv(a.output_dir/"scenario_overlap.csv",index=False)

    print("\n=== UNIFIED SIX-SCENARIO ALERT TABLE ===")
    print(summary.to_string(index=False))
    print(f"\nDevelopment transactions: {n:,}")
    print(f"Unique alerted transactions: {int(alerted.sum()):,} ({alerted.mean():.4%})")
    print(f"AML transactions among union alerts: {int((alerted&aml).sum()):,}")
    print(f"Union precision: {(alerted&aml).sum()/alerted.sum():.4%}")
    print("\n=== SCENARIO COUNT DISTRIBUTION ===")
    print(dist.to_string(index=False))
    print("\n=== TOP PAIRWISE OVERLAPS ===")
    print(overlap.head(10).to_string(index=False))
    print(f"\nSaved: {a.output_dir/'unified_alert_table.csv'}")
    print("Development-only. HOLDOUT was not accessed.")

if __name__=="__main__":
    main()
