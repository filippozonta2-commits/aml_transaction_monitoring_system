"""Train the selected XGBoost AML detector on full TRAIN and score DEVELOPMENT.

HOLDOUT is never read. This run is used to choose and document the operating
threshold before the final ML V1 freeze.
"""
from pathlib import Path
import argparse, json, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from baseline_models import enrich_country_risk, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from features import add_transaction_features, add_historical_account_features

TARGET="Is_laundering"

def prep(train,dev,risk):
    train=train.copy(); dev=dev.copy()
    train["_split"]="train"; dev["_split"]="development"
    both=pd.concat([train,dev],ignore_index=True)
    both=enrich_country_risk(both,risk)
    both=add_transaction_features(both)
    both=add_historical_account_features(both)
    return both[both["_split"].eq("train")].copy(), both[both["_split"].eq("development")].copy()

def processor():
    return ColumnTransformer([
        ("cat",Pipeline([
            ("imp",SimpleImputer(strategy="most_frequent")),
            ("oh",OneHotEncoder(handle_unknown="ignore"))
        ]),CATEGORICAL_FEATURES),
        ("num",Pipeline([
            ("imp",SimpleImputer(strategy="median"))
        ]),NUMERICAL_FEATURES),
    ])

def operating_points(y,p,rates):
    order=np.argsort(-p); n=len(y); total=int(y.sum()); rows=[]
    for rate in rates:
        k=max(1,int(np.ceil(n*rate)))
        idx=order[:k]; hits=int(y.iloc[idx].sum())
        rows.append({
            "alert_rate":rate,"alerts":k,"aml_hits":hits,
            "precision":hits/k,"recall":hits/total,
            "threshold":float(p[idx[-1]])
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    ap.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    ap.add_argument("--country-risk",type=Path,default=Path("data/country_risk.csv"))
    ap.add_argument("--out",type=Path,default=Path("results/ml_detection/full_train_development"))
    a=ap.parse_args()

    print("Loading full TRAIN + DEVELOPMENT...")
    tr=pd.read_csv(a.train); dv=pd.read_csv(a.development); risk=pd.read_csv(a.country_risk)
    tr,dv=prep(tr,dv,risk)
    feats=CATEGORICAL_FEATURES+NUMERICAL_FEATURES
    Xtr=tr[feats]; ytr=tr[TARGET].astype(int)
    Xdv=dv[feats]; ydv=dv[TARGET].astype(int)
    w=max(ytr.eq(0).sum()/max(ytr.eq(1).sum(),1),1.0)

    model=XGBClassifier(
        n_estimators=120,max_depth=5,learning_rate=.10,
        subsample=.8,colsample_bytree=.8,tree_method="hist",
        scale_pos_weight=w,n_jobs=-1,random_state=42,eval_metric="aucpr"
    )
    pipe=Pipeline([("preprocessor",processor()),("classifier",model)])
    print(f"Training selected XGBoost on FULL TRAIN: {len(tr):,} rows | AML positives: {int(ytr.sum()):,}")
    pipe.fit(Xtr,ytr)
    print("Scoring full DEVELOPMENT...")
    p=pipe.predict_proba(Xdv)[:,1]

    roc=roc_auc_score(ydv,p); pr=average_precision_score(ydv,p)
    rates=[.005,.01,.02,.04,.06,.0694,.08]
    ops=operating_points(ydv,p,rates)

    a.out.mkdir(parents=True,exist_ok=True)
    joblib.dump(pipe,a.out/"xgboost_full_train.joblib")
    ops.to_csv(a.out/"operating_points.csv",index=False)
    pd.DataFrame({"development_row_id":dv.index,TARGET:ydv,"ml_probability":p}).to_csv(a.out/"development_scores.csv",index=False)
    with open(a.out/"training_contract.json","w") as f:
        json.dump({
            "model":"XGBoost","train_rows":len(tr),"train_aml_positives":int(ytr.sum()),
            "development_rows":len(dv),"development_aml_positives":int(ydv.sum()),
            "roc_auc":roc,"pr_auc":pr,"features":feats,
            "holdout_accessed":False,"threshold_frozen":False
        },f,indent=2)

    print()
    print("=== SELECTED XGBOOST — FULL TRAIN -> DEVELOPMENT ===")
    print(f"ROC-AUC: {roc:.6f} | PR-AUC: {pr:.6f}")
    print()
    print("=== OPERATING POINTS ===")
    print(ops.to_string(index=False))
    print()
    print("Model fitted and DEVELOPMENT scored. HOLDOUT NOT accessed.")
    print("Threshold is NOT frozen yet; choose it from these DEVELOPMENT results.")
    print("Saved:",a.out)

if __name__=="__main__":
    main()
