"""Benchmark Logistic Regression vs XGBoost for DEVELOPMENT case prioritization.

Uses the frozen chronological split and reports both raw temporal validation and
the stricter new-account (account-purged) validation. HOLDOUT is not accessed.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

DROP_REDUNDANT={"amount_per_alert","payment_share_Cross-border","total_alert_amount",
                "scenario_events","multi_scenario_flag","currency_mismatch_rate"}
META={"CASE_ID","account_id","case_start","case_end","scenario_list","case_status","case_policy"}

def ranking(y,p,k):
    k=min(k,len(y))
    idx=np.argsort(-p)[:k]; yy=np.asarray(y)[idx]
    return yy.mean(), yy.sum()/np.asarray(y).sum() if np.asarray(y).sum() else np.nan

def metrics(name,y,p):
    out={"model":name,"cases":len(y),"positives":int(np.sum(y)),
         "roc_auc":roc_auc_score(y,p),"pr_auc":average_precision_score(y,p)}
    for k in [100,250,500,1000]:
        if len(y)>=k:
            pr,rc=ranking(y,p,k); out[f"precision_at_{k}"]=pr; out[f"recall_at_{k}"]=rc
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",type=Path,default=Path("results/case_features/case_features_development.csv"))
    ap.add_argument("--labels",type=Path,default=Path("results/investigator_queue/case_evaluation_labels.csv"))
    ap.add_argument("--out",type=Path,default=Path("results/ml_case_benchmark"))
    a=ap.parse_args()
    x=pd.read_csv(a.features); y=pd.read_csv(a.labels)[["CASE_ID","aml_positive"]]
    d=x.merge(y,on="CASE_ID",validate="one_to_one")
    d["case_start"]=pd.to_datetime(d["case_start"])
    d=d.sort_values(["case_start","CASE_ID"]).reset_index(drop=True)
    cut=max(1,min(len(d)-1,int(len(d)*.70))); cutoff=d.loc[cut,"case_start"]
    tr=d[d.case_start<cutoff].copy(); va=d[d.case_start>=cutoff].copy()
    vap=va[~va.account_id.astype(str).isin(set(tr.account_id.astype(str)))].copy()

    features=[c for c in d.columns if c not in META|{"aml_positive"} and
              pd.api.types.is_numeric_dtype(d[c]) and c not in DROP_REDUNDANT]
    if "account_id" in features: features.remove("account_id")

    Xtr=tr[features]; yt=tr.aml_positive.astype(int)
    scale=max((len(yt)-yt.sum())/max(yt.sum(),1),1.0)
    models={
      "logistic_regression":Pipeline([("imputer",SimpleImputer(strategy="median")),
                                      ("scale",StandardScaler()),
                                      ("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=42))]),
      "xgboost":XGBClassifier(n_estimators=350,max_depth=4,learning_rate=.05,subsample=.8,
                              colsample_bytree=.8,min_child_weight=3,reg_lambda=1.0,
                              objective="binary:logistic",eval_metric="logloss",
                              scale_pos_weight=scale,random_state=42,n_jobs=-1)
    }
    rows=[]; scores=[]
    for mn,m in models.items():
        print(f"Training {mn} on {len(tr):,} cases...")
        m.fit(Xtr,yt)
        for sn,z in [("validation_raw",va),("validation_account_purged",vap)]:
            p=m.predict_proba(z[features])[:,1]
            r=metrics(mn,z.aml_positive.astype(int).values,p); r["split"]=sn; rows.append(r)
            scores.append(pd.DataFrame({"CASE_ID":z.CASE_ID,"model":mn,"split":sn,
                                        "risk_score":p,"aml_positive":z.aml_positive.values}))
    res=pd.DataFrame(rows)
    a.out.mkdir(parents=True,exist_ok=True)
    res.to_csv(a.out/"model_benchmark.csv",index=False)
    pd.concat(scores,ignore_index=True).to_csv(a.out/"validation_scores.csv",index=False)
    pd.DataFrame({"feature":features}).to_csv(a.out/"model_features.csv",index=False)

    print("\n=== CASE ML BENCHMARK ===")
    print(res.to_string(index=False))
    print(f"\nTemporal cutoff: {cutoff}")
    print(f"Predictors used: {len(features)}")
    print("Dropped redundant predictors:",", ".join(sorted(DROP_REDUNDANT)))
    print("No threshold selected; models are evaluated as prioritization/ranking models.")
    print("Development-only. HOLDOUT was not accessed.")
    print(f"Saved: {a.out}")

if __name__=="__main__": main()
