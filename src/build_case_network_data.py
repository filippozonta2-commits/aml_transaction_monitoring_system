"""Build enriched case-to-transaction lineage for dashboard/network views.

Presentation artifact only. Does not recalculate scenarios, cases, model scores,
or AML evaluation metrics.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from country_normalization import country_key

TX_COLS=["Date","Time","Sender_account","Receiver_account","Amount","Payment_type","Payment_currency",
         "Received_currency","Sender_bank_location","Receiver_bank_location"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--holdout",type=Path,default=Path("data/temporal/SAML-D_holdout.csv"))
    p.add_argument("--lineage",type=Path,default=Path("results/holdout/case_transaction_lineage.csv"))
    p.add_argument("--out",type=Path,default=Path("results/dashboard/case_transaction_network.csv"))
    p.add_argument("--queue",type=Path,default=Path("results/dashboard/ranked_investigator_queue.csv"))
    p.add_argument("--summary",type=Path,default=Path("results/dashboard/network_complexity.csv"))
    a=p.parse_args()
    links=pd.read_csv(a.lineage)
    q=pd.read_csv(a.queue)
    tx=pd.read_csv(a.holdout,usecols=TX_COLS)
    tx["holdout_row_id"]=np.arange(len(tx),dtype="int64")
    tx["ts"]=pd.to_datetime(tx["Date"].astype(str)+" "+tx["Time"].astype(str),errors="coerce")
    alerted=set(links["holdout_row_id"].astype(int))
    wanted=set(q["account_id"].astype(str))
    z=tx[tx["Sender_account"].astype(str).isin(wanted)|tx["Receiver_account"].astype(str).isin(wanted)].copy()
    z["account_id"]=np.where(z["Sender_account"].astype(str).isin(wanted),z["Sender_account"],z["Receiver_account"])
    z=z.merge(q[["CASE_ID","account_id","case_start","case_end"]],on="account_id",how="inner")
    z["case_start"]=pd.to_datetime(z["case_start"]); z["case_end"]=pd.to_datetime(z["case_end"])
    z=z[z["ts"].between(z["case_start"]-pd.Timedelta(hours=72),z["case_end"]+pd.Timedelta(hours=72))].copy()
    z["is_alerted_transaction"]=z["holdout_row_id"].isin(alerted).astype("int8")
    scen=links.groupby("holdout_row_id")["scenario"].agg(lambda s:" | ".join(sorted(set(s.astype(str)))))
    z["scenario"]=z["holdout_row_id"].map(scen).fillna("CONTEXT")
    z["sender_iso2"]=z["Sender_bank_location"].map(country_key)
    z["receiver_iso2"]=z["Receiver_bank_location"].map(country_key)
    z["cross_border"]=(z["sender_iso2"]!=z["receiver_iso2"]).astype("int8")
    z["currency_mismatch"]=(z["Payment_currency"]!=z["Received_currency"]).astype("int8")
    z["counterparty_id"]=np.where(
        z["Sender_account"].astype(str).eq(z["account_id"].astype(str)),
        z["Receiver_account"],z["Sender_account"])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    z.to_csv(a.out,index=False)
    sm=(z.groupby(["CASE_ID","account_id"]).agg(network_transactions=("holdout_row_id","count"),unique_counterparties=("counterparty_id","nunique"),cross_border_transactions=("cross_border","sum"),alerted_transactions=("is_alerted_transaction","sum")).reset_index().sort_values(["unique_counterparties","network_transactions"],ascending=False))
    sm.to_csv(a.summary,index=False)
    print("\n=== FULL CASE TRANSACTION NETWORK MART ===")
    print(f"Scenario-event rows: {len(z):,}")
    print(f"Cases: {z.CASE_ID.nunique():,}")
    print(f"Unique transaction links: {z[['CASE_ID','holdout_row_id']].drop_duplicates().shape[0]:,}")
    print(f"Cross-border rows: {int(z.cross_border.sum()):,}")
    print(f"ISO-2 sender coverage: {z.sender_iso2.notna().mean():.2%}")
    print(f"ISO-2 receiver coverage: {z.receiver_iso2.notna().mean():.2%}")
    print("\n=== TOP TEST CASES BY NETWORK COMPLEXITY ===")
    print(sm.head(15).to_string(index=False))
    print(f"Saved: {a.out}")
    print(f"Saved: {a.summary}")
if __name__=="__main__": main()
