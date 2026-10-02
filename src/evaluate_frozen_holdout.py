"""Final single-shot HOLDOUT evaluation of the frozen case-prioritization model.

IMPORTANT: This script assumes HOLDOUT case features/labels were materialized by
the exact frozen six-scenario -> 72h case -> feature pipeline. It never tunes
features, thresholds, case policy, or hyperparameters from HOLDOUT results.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from case_model_frozen import DROP_REDUNDANT, MODEL_PARAMS, RANKING_K

META={"CASE_ID","account_id","case_start","case_end","scenario_list","case_status","case_policy"}

def topk(y,p,k):
    k=min(k,len(y)); idx=np.argsort(-p)[:k]; yy=np.asarray(y)[idx]
    return float(yy.mean()), float(yy.sum()/np.asarray(y).sum()) if np.asarray(y).sum() else np.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--development-features",type=Path,default=Path("results/case_features/case_features_development.csv"))
    ap.add_argument("--development-labels",type=Path,default=Path("results/investigator_queue/case_evaluation_labels.csv"))
    ap.add_argument("--holdout-features",type=Path,default=Path("results/holdout/case_features_holdout.csv"))
    ap.add_argument("--holdout-labels",type=Path,default=Path("results/holdout/case_evaluation_labels.csv"))
    ap.add_argument("--out",type=Path,default=Path("results/final_holdout"))
    a=ap.parse_args()

    for p in [a.development_features,a.development_labels,a.holdout_features,a.holdout_labels]:
        if not p.exists(): raise FileNotFoundError(f"Required frozen-pipeline artifact not found: {p}")

    dev=pd.read_csv(a.development_features).merge(
        pd.read_csv(a.development_labels)[["CASE_ID","aml_positive"]],on="CASE_ID",validate="one_to_one")
    ho=pd.read_csv(a.holdout_features).merge(
        pd.read_csv(a.holdout_labels)[["CASE_ID","aml_positive"]],on="CASE_ID",validate="one_to_one")

    features=[c for c in dev.columns if c not in META|{"aml_positive"}|DROP_REDUNDANT
              and pd.api.types.is_numeric_dtype(dev[c]) and c!="account_id"]
    missing=[c for c in features if c not in ho.columns]
    if missing: raise RuntimeError(f"HOLDOUT feature contract mismatch; missing: {missing}")
    if dev[features].isna().any().any() or ho[features].isna().any().any():
        raise RuntimeError("Missing values found in frozen model features.")
    if not set(ho.aml_positive.dropna().unique()).issubset({0,1}):
        raise RuntimeError("Invalid HOLDOUT labels.")

    y=dev.aml_positive.astype(int)
    scale=max((len(y)-y.sum())/max(y.sum(),1),1.0)
    params=dict(MODEL_PARAMS); params["scale_pos_weight"]=scale
    model=XGBClassifier(**params)
    print(f"Training frozen XGBoost on ALL DEVELOPMENT cases: {len(dev):,}...")
    model.fit(dev[features],y)

    yh=ho.aml_positive.astype(int).to_numpy()
    p=model.predict_proba(ho[features])[:,1]
    prevalence=float(yh.mean())
    result={"cases":len(ho),"aml_positive":int(yh.sum()),"prevalence":prevalence,
            "roc_auc":roc_auc_score(yh,p),"pr_auc":average_precision_score(yh,p)}
    result["pr_auc_lift_vs_prevalence"]=result["pr_auc"]/prevalence if prevalence else np.nan
    for k in RANKING_K:
        if len(ho)>=k:
            pr,rc=topk(yh,p,k); result[f"precision_at_{k}"]=pr; result[f"recall_at_{k}"]=rc

    a.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([result]).to_csv(a.out/"final_holdout_metrics.csv",index=False)
    pd.DataFrame({"CASE_ID":ho.CASE_ID,"risk_score":p,"aml_positive":yh}).sort_values(
        "risk_score",ascending=False).to_csv(a.out/"final_holdout_scores.csv",index=False)
    pd.DataFrame({"feature":features}).to_csv(a.out/"frozen_model_features.csv",index=False)

    print("\n=== FINAL SINGLE-SHOT HOLDOUT EVALUATION ===")
    for k,v in result.items():
        if isinstance(v,float): print(f"{k}: {v:.6f}")
        else: print(f"{k}: {v:,}")
    print(f"Frozen predictors: {len(features)}")
    print("No HOLDOUT-driven tuning performed.")
    print("These are final out-of-sample results; do not change the frozen pipeline from them.")
    print(f"Saved: {a.out}")

if __name__=="__main__": main()
