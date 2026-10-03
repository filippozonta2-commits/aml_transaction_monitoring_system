"""Compare Frozen Rule-Based V3 vs Frozen ML Detection V1 on HOLDOUT.

Pure evaluation/diagnostics. Neither detector is changed. Produces overlap
counts for all transactions and for AML positives, plus a post-hoc union
diagnostic explicitly NOT designated as a frozen production system.
"""
from pathlib import Path
import argparse, json, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rule",type=Path,default=Path("results/holdout_v3/non_deposit_send.csv"))
    ap.add_argument("--ml",type=Path,default=Path("results/ml_detection/holdout_v1/ml_holdout_scores.csv"))
    ap.add_argument("--out",type=Path,default=Path("results/comparison_rule_v3_vs_ml_v1"))
    a=ap.parse_args()

    rule=pd.read_csv(a.rule)
    ml=pd.read_csv(a.ml)

    # Frozen V3 transaction-level portfolio: OR across scenario flags present in the
    # materialized rule output. Exclude identifiers, labels and non-scenario columns.
    scenario_cols=[c for c in rule.columns if c.startswith("SCN_")]
    if not scenario_cols:
        raise ValueError("No SCN_* columns found in frozen V3 rule materialization.")
    rule_flag=rule[scenario_cols].fillna(0).astype(bool).any(axis=1)

    if len(rule_flag)!=len(ml):
        raise ValueError(f"Row-count mismatch: rule={len(rule_flag):,}, ml={len(ml):,}")

    ml_flag=ml["ML_ALERT_V1"].astype(bool)
    y=ml["Is_laundering"].astype(int).astype(bool)

    group=pd.Series("Missed by both",index=ml.index)
    group[rule_flag & ~ml_flag]="Rule only"
    group[~rule_flag & ml_flag]="ML only"
    group[rule_flag & ml_flag]="Both detected"

    detail=pd.DataFrame({
        "row_id":ml["holdout_row_id"],
        "is_laundering":y.astype(int),
        "rule_v3":rule_flag.astype(int),
        "ml_v1":ml_flag.astype(int),
        "detection_group":group,
        "ml_probability":ml["ml_probability"],
    })

    order=["Both detected","ML only","Rule only","Missed by both"]
    alltab=detail["detection_group"].value_counts().reindex(order,fill_value=0)
    amltab=detail.loc[detail.is_laundering.eq(1),"detection_group"].value_counts().reindex(order,fill_value=0)
    total_aml=int(y.sum())
    summary=pd.DataFrame({
        "group":order,
        "transactions":[int(alltab[x]) for x in order],
        "aml_positives":[int(amltab[x]) for x in order],
    })
    summary["share_of_all_transactions"]=summary.transactions/len(detail)
    summary["share_of_all_aml"]=summary.aml_positives/total_aml

    union=rule_flag | ml_flag
    union_alerts=int(union.sum()); union_hits=int((union & y).sum())
    hybrid={
        "status":"POST-HOC DIAGNOSTIC ONLY — NOT FROZEN",
        "alerts":union_alerts,
        "alert_rate":union_alerts/len(detail),
        "aml_hits":union_hits,
        "precision":union_hits/union_alerts if union_alerts else 0,
        "recall":union_hits/total_aml if total_aml else 0,
    }

    a.out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(a.out/"overlap_summary.csv",index=False)
    detail.to_csv(a.out/"transaction_detection_groups.csv",index=False)
    with open(a.out/"posthoc_union_diagnostic.json","w") as f: json.dump(hybrid,f,indent=2)

    print("=== FROZEN RULE V3 vs FROZEN ML V1 — HOLDOUT OVERLAP ===")
    print(f"Transactions: {len(detail):,} | AML positives: {total_aml:,}")
    print()
    print(summary.to_string(index=False))
    print()
    print("=== POST-HOC UNION DIAGNOSTIC (NOT FROZEN) ===")
    print(f"Alerts: {union_alerts:,} | alert rate: {hybrid['alert_rate']:.4%}")
    print(f"AML hits: {union_hits:,}/{total_aml:,} | precision: {hybrid['precision']:.4%} | recall: {hybrid['recall']:.4%}")
    print()
    print("Frozen V3 and ML V1 remain unchanged. Do NOT present the union as an out-of-sample frozen model.")
    print("Saved:",a.out)

if __name__=="__main__":
    main()
