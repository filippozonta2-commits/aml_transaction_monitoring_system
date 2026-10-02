"""Build leakage-safe DEVELOPMENT case-level features for AML prioritization.

Features are derived only from information available in the transaction/case
context. AML truth remains in the separate evaluation-label table.
HOLDOUT is not accessed.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from country_normalization import country_key

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--queue-dir",type=Path,default=Path("results/investigator_queue"))
    p.add_argument("--out",type=Path,default=Path("results/case_features"))
    a=p.parse_args()

    print("Loading DEVELOPMENT + frozen case lineage...")
    tx=pd.read_csv(a.development)
    links=pd.read_csv(a.queue_dir/"case_transaction_lineage.csv")
    queue=pd.read_csv(a.queue_dir/"investigator_queue.csv")
    tx["development_row_id"]=np.arange(len(tx),dtype="int64")

    cols=["development_row_id","Sender_account","Receiver_account","Amount","Payment_type",
          "Payment_currency","Received_currency","Sender_bank_location","Receiver_bank_location"]
    z=links.merge(tx[cols],on="development_row_id",how="left",validate="many_to_one")
    if z["Amount"].isna().any(): raise RuntimeError("Case lineage failed to resolve DEVELOPMENT transactions.")

    z["cross_border"]=(z["Sender_bank_location"]!=z["Receiver_bank_location"]).astype("int8")
    z["currency_mismatch"]=(z["Payment_currency"]!=z["Received_currency"]).astype("int8")
    z["sender_iso2"]=z["Sender_bank_location"].map(country_key)
    z["receiver_iso2"]=z["Receiver_bank_location"].map(country_key)

    # Collapse duplicate scenario-events to one transaction per case before
    # transaction-level aggregates; scenario diversity already lives in queue.
    t=z.sort_values(["CASE_ID","development_row_id"]).drop_duplicates(["CASE_ID","development_row_id"])
    t["counterparty_id"]=np.where(t["Sender_account"].astype(str).eq(t["account_id"].astype(str)),
                                  t["Receiver_account"],t["Sender_account"])
    g=t.groupby("CASE_ID").agg(
        tx_amount_sum=("Amount","sum"),tx_amount_mean=("Amount","mean"),
        tx_amount_median=("Amount","median"),tx_amount_max=("Amount","max"),
        tx_amount_std=("Amount","std"),unique_counterparties=("counterparty_id","nunique"),
        payment_types=("Payment_type","nunique"),cross_border_rate=("cross_border","mean"),
        currency_mismatch_rate=("currency_mismatch","mean"),
        sender_countries=("sender_iso2","nunique"),receiver_countries=("receiver_iso2","nunique"),
    ).reset_index()
    g["tx_amount_std"]=g["tx_amount_std"].fillna(0.0)

    # Payment-channel shares, useful for explainable prioritization.
    pt=(pd.crosstab(t["CASE_ID"],t["Payment_type"],normalize="index")
        .add_prefix("payment_share_").reset_index())
    feat=(queue.merge(g,on="CASE_ID",how="left",validate="one_to_one")
          .merge(pt,on="CASE_ID",how="left",validate="one_to_one"))

    # Derived behavioral intensity; no AML labels.
    feat["alerts_per_hour"]=feat["transaction_alerts"]/(feat["duration_hours"]+1.0)
    feat["amount_per_alert"]=feat["tx_amount_sum"]/feat["transaction_alerts"].clip(lower=1)

    forbidden={"Is_laundering","Laundering_type","aml_positive","aml_event_count","aml_typologies"}
    leaked=forbidden.intersection(feat.columns)
    if leaked: raise RuntimeError(f"Label leakage detected: {sorted(leaked)}")

    a.out.mkdir(parents=True,exist_ok=True)
    feat.to_csv(a.out/"case_features_development.csv",index=False)

    print("\n=== CASE FEATURE ENGINEERING ===")
    print(f"Cases: {len(feat):,}")
    print(f"Features/columns: {len(feat.columns)}")
    print(f"Unique accounts: {feat.account_id.nunique():,}")
    print(f"Cross-border cases: {(feat.cross_border_rate.gt(0)).sum():,}")
    print(f"FX-mismatch cases: {(feat.currency_mismatch_rate.gt(0)).sum():,}")
    print(f"Multi-scenario cases: {feat.multi_scenario_flag.sum():,}")
    print(f"Missing values total: {int(feat.isna().sum().sum()):,}")
    print("Leakage check: PASS — AML truth columns absent.")
    print(f"Saved: {a.out/'case_features_development.csv'}")
    print("Development-only. HOLDOUT was not accessed.")

if __name__=="__main__": main()
