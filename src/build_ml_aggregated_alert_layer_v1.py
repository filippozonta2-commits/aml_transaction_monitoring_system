"""Aggregate Frozen ML Detection V1 transaction alerts into investigation alerts.

Operationalization only: the frozen XGBoost model and threshold 0.653366 are
unchanged. Aggregation is account-centric and uses a fixed inactivity session.
AML labels are never read or written by this layer.
"""
from pathlib import Path
import argparse, hashlib, json, numpy as np, pandas as pd

THRESHOLD=0.653366

def aid(subject,seq):
    raw=f"MLV1|AGG|{subject}|{int(seq)}".encode()
    return "ALT-ML1A-"+hashlib.sha1(raw).hexdigest()[:16].upper()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scores",type=Path,default=Path("results/ml_detection/holdout_v1/ml_holdout_scores.csv"))
    p.add_argument("--holdout",type=Path,default=Path("data/temporal/SAML-D_holdout.csv"))
    p.add_argument("--outdir",type=Path,default=Path("results/ml_case_management_v1/aggregated"))
    p.add_argument("--window-days",type=int,default=7)
    a=p.parse_args()

    scores=pd.read_csv(a.scores,usecols=["holdout_row_id","ml_probability","ML_ALERT_V1"])
    scores=scores[scores.ML_ALERT_V1.eq(1)].copy()
    # Deliberately do not read Is_laundering from the frozen score file.
    raw=pd.read_csv(a.holdout,usecols=["Date","Time","Sender_account","Receiver_account","Amount"])
    raw["holdout_row_id"]=np.arange(len(raw))
    z=scores.merge(raw,on="holdout_row_id",how="left",validate="one_to_one")
    date_s=z["Date"].astype("string").str.strip()
    time_s=z["Time"].astype("string").str.strip()
    time_s=time_s.mask(time_s.isin(["<NA>","nan","NaN","None",""]))
    combined=date_s.where(time_s.isna(),date_s+" "+time_s)
    z["ts"]=pd.to_datetime(combined,errors="coerce")
    missing=z["ts"].isna()
    if missing.any():
        z.loc[missing,"ts"]=pd.to_datetime(z.loc[missing,"Date"],errors="coerce")
    if z.ts.isna().any():
        bad=z.loc[z.ts.isna(),["holdout_row_id","Date","Time"]].head(10)
        raise ValueError("Unparseable HOLDOUT timestamps remain after Date fallback:\n"+bad.to_string(index=False))

    # Primary subject is sender account, matching transaction-originating detection grain.
    z["primary_account_id"]=z.Sender_account.astype(str)
    z["transaction_id"]=z.holdout_row_id.map(lambda i:f"TXN-HO-{int(i):010d}")
    z=z.sort_values(["primary_account_id","ts","transaction_id"]).copy()
    prev=z.groupby("primary_account_id")["ts"].shift()
    gap=(z.ts-prev).dt.total_seconds()/86400
    z["_seq"]=(prev.isna()|gap.gt(a.window_days)).groupby(z.primary_account_id).cumsum().astype(int)
    z["alert_id"]=[aid(s,q) for s,q in zip(z.primary_account_id,z._seq)]

    g=z.groupby("alert_id",sort=False)
    agg=g.agg(
        primary_account_id=("primary_account_id","first"),
        alert_created_at=("ts","min"),last_trigger_at=("ts","max"),
        transaction_count=("transaction_id","nunique"),
        alert_amount=("Amount","sum"),
        ml_probability_max=("ml_probability","max"),
        ml_probability_mean=("ml_probability","mean"),
    ).reset_index()
    agg["detection_engine"]="ML_V1"; agg["model_version"]="ML_DETECTION_V1"
    agg["threshold"]=THRESHOLD; agg["aggregation_window_days"]=a.window_days
    agg["alert_status"]="NEW"; agg["priority"]="UNASSIGNED"
    agg["trigger_reason"]=agg.apply(lambda r:
        f"Frozen ML V1; {int(r.transaction_count)} triggering transaction(s); "
        f"max probability={r.ml_probability_max:.4f}; {a.window_days}d inactivity-session aggregation",axis=1)

    bridge=z[["alert_id","transaction_id","holdout_row_id","ts","ml_probability"]].copy()
    bridge=bridge.rename(columns={"ts":"linked_at"})
    bridge["contribution_role"]="TRIGGER"

    a.outdir.mkdir(parents=True,exist_ok=True)
    agg.to_csv(a.outdir/"alert.csv",index=False)
    bridge.to_csv(a.outdir/"alert_transaction.csv",index=False)
    manifest={
      "detector":"Frozen ML Detection V1","threshold":THRESHOLD,
      "input_transaction_alerts":int(len(z)),"aggregated_alerts":int(len(agg)),
      "reduction_pct":float(1-len(agg)/len(z)),
      "unique_accounts":int(agg.primary_account_id.nunique()),
      "aggregation_window_days":a.window_days,
      "aml_labels_used":False,
      "note":"Operational aggregation only. Frozen ML V1 detection and HOLDOUT validation unchanged."
    }
    (a.outdir/"aggregation_manifest.json").write_text(json.dumps(manifest,indent=2))

    print("=== ML V1 AGGREGATED ALERT LAYER ===")
    print(f"Input frozen ML transaction alerts: {len(z):,}")
    print(f"Aggregated investigation alerts: {len(agg):,}")
    print(f"Alert reduction: {1-len(agg)/len(z):.2%}")
    print(f"Unique primary accounts: {agg.primary_account_id.nunique():,}")
    print()
    print("Transactions per aggregated alert:")
    print(agg.transaction_count.describe(percentiles=[.5,.75,.9,.95,.99]).to_string())
    print()
    print("ML probability max:")
    print(agg.ml_probability_max.describe(percentiles=[.5,.75,.9,.95,.99]).to_string())
    print()
    print("Operational aggregation only. No AML labels used; Frozen ML V1 unchanged.")
    print("Saved:",a.outdir)

if __name__=="__main__": main()
