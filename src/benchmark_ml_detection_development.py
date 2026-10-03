"""Benchmark transaction-level ML detectors on TRAIN -> DEVELOPMENT only.

This is the model-selection stage for the ML detection engine. HOLDOUT is never read.
Historical features are computed on TRAIN+DEVELOPMENT in chronological order so
DEVELOPMENT rows can use prior TRAIN history without seeing future observations.
"""
from pathlib import Path
import argparse, json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from baseline_models import enrich_country_risk, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from features import add_transaction_features, add_historical_account_features

TARGET="Is_laundering"

def prep(train,dev,risk):
    train=train.copy(); dev=dev.copy(); train["_split"]="train"; dev["_split"]="development"
    both=pd.concat([train,dev],ignore_index=True)
    both=enrich_country_risk(both,risk)
    both=add_transaction_features(both)
    both=add_historical_account_features(both)
    return both[both._split.eq("train")].copy(),both[both._split.eq("development")].copy()

def processor():
    return ColumnTransformer([
      ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),CATEGORICAL_FEATURES),
      ("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),NUMERICAL_FEATURES),
    ])

def metrics(y,p,name):
    return {"model":name,"roc_auc":roc_auc_score(y,p),"pr_auc":average_precision_score(y,p)}

def workload_table(y,p,name,rates):
    order=np.argsort(-p); n=len(y); rows=[]
    for rate in rates:
        k=max(1,int(np.ceil(n*rate))); idx=order[:k]; hits=int(y.iloc[idx].sum())
        rows.append({"model":name,"alert_rate":rate,"alerts":k,"aml_hits":hits,
                     "precision":hits/k,"recall":hits/int(y.sum()),
                     "threshold":float(p[idx[-1]])})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    ap.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    ap.add_argument("--country-risk",type=Path,default=Path("data/country_risk.csv"))
    ap.add_argument("--out",type=Path,default=Path("results/ml_detection/development"))
    a=ap.parse_args()
    print("Loading TRAIN and DEVELOPMENT only...")
    tr=pd.read_csv(a.train); dv=pd.read_csv(a.development); risk=pd.read_csv(a.country_risk)
    tr,dv=prep(tr,dv,risk)
    feats=CATEGORICAL_FEATURES+NUMERICAL_FEATURES; Xtr=tr[feats]; ytr=tr[TARGET].astype(int); Xdv=dv[feats]; ydv=dv[TARGET].astype(int)
    w=max((ytr.eq(0).sum()/max(ytr.eq(1).sum(),1)),1.0)
    models={
      "Logistic Regression":LogisticRegression(max_iter=500,class_weight="balanced",solver="liblinear",random_state=42),
      "Random Forest":RandomForestClassifier(n_estimators=150,max_depth=12,min_samples_leaf=40,class_weight="balanced",n_jobs=-1,random_state=42),
      "XGBoost":XGBClassifier(n_estimators=200,max_depth=6,learning_rate=.08,subsample=.8,colsample_bytree=.8,scale_pos_weight=w,n_jobs=-1,random_state=42,eval_metric="aucpr")
    }
    perf=[]; workloads=[]; probabilities={}
    for name,clf in models.items():
        print("Training",name)
        pipe=Pipeline([("preprocessor",processor()),("classifier",clf)])
        pipe.fit(Xtr,ytr); p=pipe.predict_proba(Xdv)[:,1]; probabilities[name]=p
        perf.append(metrics(ydv,p,name))
        workloads += workload_table(ydv,p,name,[.005,.01,.02,.04,.06,.08])
    perf=pd.DataFrame(perf).sort_values("pr_auc",ascending=False); work=pd.DataFrame(workloads)
    a.out.mkdir(parents=True,exist_ok=True); perf.to_csv(a.out/"model_benchmark.csv",index=False); work.to_csv(a.out/"workload_benchmark.csv",index=False)
    pd.DataFrame({"development_row_id":dv.index,"Is_laundering":ydv,**{k:v for k,v in probabilities.items()}}).to_csv(a.out/"development_scores.csv",index=False)
    with open(a.out/"feature_contract.json","w") as f: json.dump({"features":feats,"holdout_accessed":False,"historical_features":"prior-only; TRAIN history available to DEVELOPMENT"},f,indent=2)
    print("\n=== ML DETECTION BENCHMARK — DEVELOPMENT ONLY ==="); print(perf.to_string(index=False))
    print("\n=== WORKLOAD / RECALL TRADE-OFF ==="); print(work.to_string(index=False))
    print("\nNo HOLDOUT accessed. Do not run ML HOLDOUT scoring until model + threshold are frozen.")
    print("Saved:",a.out)
if __name__=="__main__":main()
