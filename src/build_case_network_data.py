"""Build enriched case-to-transaction lineage for dashboard/network views.

Presentation artifact only. Does not recalculate scenarios, cases, model scores,
or AML evaluation metrics.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from country_normalization import country_key

TX_COLS=["Sender_account","Receiver_account","Amount","Payment_type","Payment_currency",
         "Received_currency","Sender_bank_location","Receiver_bank_location"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--holdout",type=Path,default=Path("data/temporal/SAML-D_holdout.csv"))
    p.add_argument("--lineage",type=Path,default=Path("results/holdout/case_transaction_lineage.csv"))
    p.add_argument("--out",type=Path,default=Path("results/dashboard/case_transaction_network.csv"))
    a=p.parse_args()
    links=pd.read_csv(a.lineage)
    tx=pd.read_csv(a.holdout,usecols=TX_COLS)
    tx["holdout_row_id"]=np.arange(len(tx),dtype="int64")
    z=links.merge(tx,on="holdout_row_id",how="left",validate="many_to_one")
    z["sender_iso2"]=z["Sender_bank_location"].map(country_key)
    z["receiver_iso2"]=z["Receiver_bank_location"].map(country_key)
    z["cross_border"]=(z["sender_iso2"]!=z["receiver_iso2"]).astype("int8")
    z["currency_mismatch"]=(z["Payment_currency"]!=z["Received_currency"]).astype("int8")
    z["counterparty_id"]=np.where(
        z["Sender_account"].astype(str).eq(z["account_id"].astype(str)),
        z["Receiver_account"],z["Sender_account"])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    z.to_csv(a.out,index=False)
    print("\n=== CASE TRANSACTION NETWORK MART ===")
    print(f"Scenario-event rows: {len(z):,}")
    print(f"Cases: {z.CASE_ID.nunique():,}")
    print(f"Unique transaction links: {z[['CASE_ID','holdout_row_id']].drop_duplicates().shape[0]:,}")
    print(f"Cross-border rows: {int(z.cross_border.sum()):,}")
    print(f"ISO-2 sender coverage: {z.sender_iso2.notna().mean():.2%}")
    print(f"ISO-2 receiver coverage: {z.receiver_iso2.notna().mean():.2%}")
    print(f"Saved: {a.out}")
if __name__=="__main__": main()
