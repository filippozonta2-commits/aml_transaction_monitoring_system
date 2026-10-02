"""Validate DEVELOPMENT case prioritization beyond headline model metrics.

Compares XGBoost against simple operational ranking baselines, reports gain over
random prevalence, permutation importance on the strict account-purged temporal
validation set, and a compact feature-family ablation. No HOLDOUT access.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

META={"CASE_ID","account_id","case_start","case_end","scenario_list","case_status","case_policy"}
DROP={"amount_per_alert","currency_mismatch_rate","multi_scenario_flag",
      "payment_share_Cross-border","scenario_events","total_alert_amount"}
ABLATE={
 "amount":["tx_amount_sum","tx_amount_mean","tx_amount_median","tx_amount_max","tx_amount_std"],
 "volume":["transaction_alerts","alerts_per_hour"],
 "scenario":["active_scenarios"],
 "network":["unique_counterparties"],
 "geography":["cross_border_rate","sender_countries","receiver_countries"],
}

def topk(y,s,k):
    k=min(k,len(y)); idx=np.argsort(-np.asarray(s))[:k]; yy=np.asarray(y)[idx]
    return float(yy.mean()),float(yy.sum()/np.asarray(y).sum())

def evaluate(name,y,s):
    r={"ranking":name,"roc_auc":roc_auc_score(y,s),"pr_auc":average_precision_score(y,s)}
    for k in [100,250,500]:
        p,rc=topk(y,s,k); r[f"precision_at_{k}"]=p; r[f"recall_at_{k}"]=rc
    return r

def fit_xgb(tr,features):
    y=tr.aml_positive.astype(int)
    scale=max((len(y)-y.sum())/max(y.sum(),1),1)
    m=XGBClassifier(n_estimators=350,max_depth=4,learning_rate=.05,subsample=.8,
        colsample_bytree=.8,min_child_weight=3,reg_lambda=1.0,
        objective="binary:logistic",eval_metric="logloss",scale_pos_weight=scale,
        random_state=42,n_jobs=-1)
    m.fit(tr[features],y); return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",type=Path,default=Path("results/case_features/case_features_development.csv"))
    ap.add_argument("--labels",type=Path,default=Path("results/investigator_queue/case_evaluation_labels.csv"))
    ap.add_argument("--out",type=Path,default=Path("results/ml_case_validation"))
    a=ap.parse_args()
    d=pd.read_csv(a.features).merge(pd.read_csv(a.labels)[["CASE_ID","aml_positive"]],on="CASE_ID",validate="one_to_one")
    d["case_start"]=pd.to_datetime(d.case_start); d=d.sort_values(["case_start","CASE_ID"]).reset_index(drop=True)
    cutoff=d.loc[int(len(d)*.70),"case_start"]
    tr=d[d.case_start<cutoff].copy(); raw=d[d.case_start>=cutoff].copy()
    va=raw[~raw.account_id.astype(str).isin(set(tr.account_id.astype(str)))].copy()
    features=[c for c in d.columns if c not in META|{"aml_positive"}|DROP and pd.api.types.is_numeric_dtype(d[c])]
    features=[c for c in features if c!="account_id"]

    model=fit_xgb(tr,features); score=model.predict_proba(va[features])[:,1]
    rows=[evaluate("xgboost",va.aml_positive,score)]
    baselines={
      "transaction_alerts":va.transaction_alerts,
      "tx_amount_sum":va.tx_amount_sum,
      "active_scenarios":va.active_scenarios,
      "cross_border_rate":va.cross_border_rate,
      "unique_counterparties":va.unique_counterparties,
    }
    for n,s in baselines.items(): rows.append(evaluate("baseline_"+n,va.aml_positive,s))
    res=pd.DataFrame(rows)
    prevalence=va.aml_positive.mean()
    res["pr_auc_lift_vs_prevalence"]=res.pr_auc/prevalence

    perm=permutation_importance(model,va[features],va.aml_positive,n_repeats=8,
                                scoring="average_precision",random_state=42,n_jobs=-1)
    imp=pd.DataFrame({"feature":features,"pr_auc_importance_mean":perm.importances_mean,
                      "pr_auc_importance_std":perm.importances_std}).sort_values("pr_auc_importance_mean",ascending=False)

    abl=[]
    base=evaluate("all_features",va.aml_positive,score)
    abl.append({"variant":"all_features","n_features":len(features),"pr_auc":base["pr_auc"],
                "roc_auc":base["roc_auc"],"precision_at_250":base["precision_at_250"],"recall_at_250":base["recall_at_250"]})
    for family,cols in ABLATE.items():
        fs=[f for f in features if f not in cols]
        m=fit_xgb(tr,fs); s=m.predict_proba(va[fs])[:,1]; r=evaluate("x",va.aml_positive,s)
        abl.append({"variant":"without_"+family,"n_features":len(fs),"pr_auc":r["pr_auc"],
                    "roc_auc":r["roc_auc"],"precision_at_250":r["precision_at_250"],"recall_at_250":r["recall_at_250"]})
    abl=pd.DataFrame(abl)

    a.out.mkdir(parents=True,exist_ok=True)
    res.to_csv(a.out/"baseline_comparison.csv",index=False)
    imp.to_csv(a.out/"permutation_importance.csv",index=False)
    abl.to_csv(a.out/"feature_family_ablation.csv",index=False)

    print("\n=== XGBOOST VS SIMPLE RANKING BASELINES — ACCOUNT-PURGED VALIDATION ===")
    print(f"Cases: {len(va):,} | AML-positive: {int(va.aml_positive.sum()):,} | prevalence: {prevalence:.2%}")
    print(res.to_string(index=False))
    print("\n=== TOP PERMUTATION IMPORTANCE (PR-AUC) ===")
    print(imp.head(15).to_string(index=False))
    print("\n=== FEATURE-FAMILY ABLATION ===")
    print(abl.to_string(index=False))
    print("\nInterpretation only after these diagnostics; no threshold/model freeze here.")
    print("Development-only. HOLDOUT was not accessed.")
    print(f"Saved: {a.out}")

if __name__=="__main__": main()
