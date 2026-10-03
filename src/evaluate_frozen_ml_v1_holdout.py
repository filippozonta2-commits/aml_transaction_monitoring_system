"""Score Frozen ML Detection V1 on HOLDOUT exactly once. No tuning."""
from pathlib import Path
import argparse, json, joblib, pandas as pd
from baseline_models import enrich_country_risk, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from features import add_transaction_features, add_historical_account_features

THRESHOLD=0.653366
TARGET="Is_laundering"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    ap.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    ap.add_argument("--holdout",type=Path,default=Path("data/temporal/SAML-D_holdout.csv"))
    ap.add_argument("--country-risk",type=Path,default=Path("data/country_risk.csv"))
    ap.add_argument("--model",type=Path,default=Path("results/ml_detection/full_train_development/xgboost_full_train.joblib"))
    ap.add_argument("--out",type=Path,default=Path("results/ml_detection/holdout_v1"))
    a=ap.parse_args()

    print("Loading TRAIN + DEVELOPMENT + HOLDOUT for chronological feature construction...")
    tr=pd.read_csv(a.train); dv=pd.read_csv(a.development); ho=pd.read_csv(a.holdout)
    risk=pd.read_csv(a.country_risk)
    tr["_split"]="train"; dv["_split"]="development"; ho["_split"]="holdout"
    # Labels are carried only for final evaluation; feature functions do not consume TARGET.
    allx=pd.concat([tr,dv,ho],ignore_index=True)
    allx=enrich_country_risk(allx,risk)
    allx=add_transaction_features(allx)
    allx=add_historical_account_features(allx)
    h=allx[allx["_split"].eq("holdout")].copy()

    pipe=joblib.load(a.model)
    feats=CATEGORICAL_FEATURES+NUMERICAL_FEATURES
    print(f"Scoring HOLDOUT: {len(h):,} transactions | frozen threshold={THRESHOLD:.6f}")
    p=pipe.predict_proba(h[feats])[:,1]
    flag=p>=THRESHOLD
    y=h[TARGET].astype(int)
    alerts=int(flag.sum()); hits=int(y[flag].sum()); total=int(y.sum())
    precision=hits/alerts if alerts else 0.0
    recall=hits/total if total else 0.0
    alert_rate=alerts/len(h)

    a.out.mkdir(parents=True,exist_ok=True)
    out=pd.DataFrame({"holdout_row_id":h.index,"ml_probability":p,"ML_ALERT_V1":flag.astype(int),TARGET:y.values})
    out.to_csv(a.out/"ml_holdout_scores.csv",index=False)
    metrics={"transactions":len(h),"aml_positives":total,"threshold":THRESHOLD,
             "alerts":alerts,"alert_rate":alert_rate,"aml_hits":hits,
             "precision":precision,"recall":recall}
    with open(a.out/"metrics.json","w") as f: json.dump(metrics,f,indent=2)

    print()
    print("=== FROZEN ML DETECTION V1 — HOLDOUT ===")
    print(f"Transactions: {len(h):,} | AML positives: {total:,}")
    print(f"Threshold: {THRESHOLD:.6f}")
    print(f"Alerts: {alerts:,} | alert rate: {alert_rate:.4%}")
    print(f"AML hits: {hits:,}/{total:,} | precision: {precision:.4%} | recall: {recall:.4%}")
    print()
    print("Frozen V1 evaluation only. DO NOT tune model or threshold on these HOLDOUT results.")
    print("Saved:",a.out)

if __name__=="__main__":
    main()
