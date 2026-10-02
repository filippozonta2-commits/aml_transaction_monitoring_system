"""Audit DEVELOPMENT case features and define a leakage-aware temporal ML split.

Labels are joined only for diagnostics/evaluation. No model is trained here.
HOLDOUT is not accessed.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

META={"CASE_ID","account_id","case_start","case_end","scenario_list","case_status","case_policy"}
LABEL={"aml_positive","aml_event_count","aml_typologies"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=Path,default=Path("results/case_features/case_features_development.csv"))
    p.add_argument("--labels",type=Path,default=Path("results/investigator_queue/case_evaluation_labels.csv"))
    p.add_argument("--out",type=Path,default=Path("results/ml_case_audit"))
    a=p.parse_args()
    x=pd.read_csv(a.features); y=pd.read_csv(a.labels)
    x["case_start"]=pd.to_datetime(x["case_start"],errors="coerce")
    x["case_end"]=pd.to_datetime(x["case_end"],errors="coerce")
    d=x.merge(y[["CASE_ID","aml_positive"]],on="CASE_ID",how="left",validate="one_to_one")
    if d["aml_positive"].isna().any(): raise RuntimeError("Missing evaluation labels.")
    if LABEL.intersection(x.columns): raise RuntimeError("AML label leakage found in feature file.")

    # Operational categorical metadata is excluded from the initial numeric benchmark.
    numeric=[c for c in x.columns if c not in META and pd.api.types.is_numeric_dtype(x[c])]
    # account_id is intentionally excluded even if numeric-looking: identity is not a behavioral feature.
    numeric=[c for c in numeric if c!="account_id"]

    rows=[]
    for c in numeric:
        s=d[c]
        rows.append({"feature":c,"nunique":s.nunique(),"missing":int(s.isna().sum()),
                     "mean":s.mean(),"std":s.std(),"min":s.min(),"p50":s.median(),
                     "p95":s.quantile(.95),"max":s.max(),
                     "mean_aml":d.loc[d.aml_positive.eq(1),c].mean(),
                     "mean_non_aml":d.loc[d.aml_positive.eq(0),c].mean()})
    audit=pd.DataFrame(rows)
    audit["near_constant"]=audit["nunique"].le(1)

    corr=d[numeric].corr().abs()
    pairs=[]
    for i,c1 in enumerate(numeric):
        for c2 in numeric[i+1:]:
            v=corr.loc[c1,c2]
            if pd.notna(v) and v>=.95:
                pairs.append({"feature_a":c1,"feature_b":c2,"abs_correlation":v})
    pairs=pd.DataFrame(pairs,columns=["feature_a","feature_b","abs_correlation"]).sort_values("abs_correlation",ascending=False)

    # Chronological 70/30 split by case start. Purge accounts crossing the boundary
    # from validation to make the benchmark stricter and avoid same-account leakage.
    ordered=d.sort_values(["case_start","CASE_ID"]).reset_index(drop=True)
    cut=max(1,min(len(ordered)-1,int(len(ordered)*.70)))
    cutoff=ordered.loc[cut,"case_start"]
    train=ordered[ordered.case_start<cutoff].copy()
    valid=ordered[ordered.case_start>=cutoff].copy()
    train_accounts=set(train.account_id.astype(str))
    overlap=valid.account_id.astype(str).isin(train_accounts)
    valid_purged=valid.loc[~overlap].copy()

    split=pd.DataFrame([
        {"split":"train","cases":len(train),"aml_positive":int(train.aml_positive.sum()),
         "aml_rate":train.aml_positive.mean(),"start":train.case_start.min(),"end":train.case_start.max()},
        {"split":"validation_raw","cases":len(valid),"aml_positive":int(valid.aml_positive.sum()),
         "aml_rate":valid.aml_positive.mean(),"start":valid.case_start.min(),"end":valid.case_start.max()},
        {"split":"validation_account_purged","cases":len(valid_purged),"aml_positive":int(valid_purged.aml_positive.sum()),
         "aml_rate":valid_purged.aml_positive.mean() if len(valid_purged) else np.nan,
         "start":valid_purged.case_start.min() if len(valid_purged) else pd.NaT,
         "end":valid_purged.case_start.max() if len(valid_purged) else pd.NaT},
    ])

    a.out.mkdir(parents=True,exist_ok=True)
    audit.to_csv(a.out/"feature_audit.csv",index=False)
    pairs.to_csv(a.out/"high_correlation_pairs.csv",index=False)
    split.to_csv(a.out/"temporal_split_summary.csv",index=False)
    pd.DataFrame({"feature":numeric}).to_csv(a.out/"initial_numeric_features.csv",index=False)

    print("\n=== CASE FEATURE AUDIT ===")
    print(f"Cases: {len(d):,} | AML-positive: {int(d.aml_positive.sum()):,} ({d.aml_positive.mean():.2%})")
    print(f"Initial numeric predictors: {len(numeric)}")
    print(f"Near-constant predictors: {int(audit.near_constant.sum())}")
    print(f"Highly correlated pairs (|r| >= 0.95): {len(pairs)}")
    if len(pairs): print(pairs.head(15).to_string(index=False))
    print("\n=== TEMPORAL SPLIT DESIGN ===")
    print(split.to_string(index=False))
    print(f"Validation cases removed by account purge: {int(overlap.sum()):,}")
    print(f"Temporal cutoff: {cutoff}")
    print("\nNo model trained. AML labels used only for diagnostics/split reporting.")
    print(f"Saved: {a.out}")
    print("Development-only. HOLDOUT was not accessed.")

if __name__=="__main__": main()
