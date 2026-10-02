"""Compare generic vs Cash-Deposit-aware Smurfing rules on DEVELOPMENT.

TRAIN is used only as historical warm-up. HOLDOUT is never accessed.
The goal is to quantify whether payment-method segmentation improves alert
efficiency without sacrificing Smurfing coverage.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load(path, period):
    usecols=["Time","Date","Sender_account","Receiver_account","Amount",
             "Payment_type","Sender_bank_location","Receiver_bank_location",
             "Payment_currency","Received_currency","Is_laundering",
             "Laundering_type"]
    d=pd.read_csv(path,usecols=usecols)
    d["ts"]=pd.to_datetime(
        pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")
        +" "+d["Time"].astype(str),errors="coerce")
    d["Period"]=period
    d["cross"]=(d["Sender_bank_location"]!=d["Receiver_bank_location"]).astype("int8")
    d["fx"]=(d["Payment_currency"]!=d["Received_currency"]).astype("int8")
    return d


def aggregate(data,start,days):
    x=data.copy()
    x["window_id"]=np.floor(
        (x["ts"]-start).dt.total_seconds()/86400/days
    ).astype("int32")
    x["dev"]=x["Period"].eq("development").astype("int8")
    x["dev_aml"]=(x["Period"].eq("development")&
                  x["Is_laundering"].eq(1)).astype("int8")
    x["dev_smurf"]=(x["Period"].eq("development")&
                    x["Is_laundering"].eq(1)&
                    x["Laundering_type"].eq("Smurfing")).astype("int8")

    base=x.groupby(["window_id","Sender_account"],observed=True).agg(
        tx_count=("Amount","size"),
        aggregate_amount=("Amount","sum"),
        median_amount=("Amount","median"),
        unique_receivers=("Receiver_account","nunique"),
        cross_rate=("cross","mean"),
        fx_rate=("fx","mean"),
        dev_transactions=("dev","sum"),
        dev_aml=("dev_aml","sum"),
        dev_smurf=("dev_smurf","sum"),
    ).reset_index()

    cash=x[x["Payment_type"].eq("Cash Deposit")].groupby(
        ["window_id","Sender_account"],observed=True
    ).agg(
        cash_tx_count=("Amount","size"),
        cash_aggregate_amount=("Amount","sum"),
        cash_median_amount=("Amount","median"),
        cash_unique_receivers=("Receiver_account","nunique"),
        cash_cross_rate=("cross","mean"),
        cash_fx_rate=("fx","mean"),
        cash_dev_transactions=("dev","sum"),
        cash_dev_aml=("dev_aml","sum"),
        cash_dev_smurf=("dev_smurf","sum"),
    ).reset_index()

    return base.merge(cash,on=["window_id","Sender_account"],how="left").fillna(0)


def score(g,mask,prefix):
    h=g.loc[mask]
    total_smurf=int(g["dev_smurf"].sum())
    target_accounts=g.loc[g["dev_smurf"].gt(0),"Sender_account"].nunique()
    hit_accounts=h.loc[h["dev_smurf"].gt(0),"Sender_account"].nunique()
    trig=int(h[f"{prefix}dev_transactions"].sum())
    aml=int(h[f"{prefix}dev_aml"].sum())
    smurf=int(h[f"{prefix}dev_smurf"].sum())
    return dict(
        Triggered_transactions=trig,
        Trigger_rate=trig/int(g["dev_transactions"].sum()),
        AML_cases=aml,
        Precision=aml/trig if trig else np.nan,
        Smurfing_hits=smurf,
        Smurfing_recall=smurf/total_smurf if total_smurf else np.nan,
        Smurfing_hit_accounts=hit_accounts,
        Smurfing_account_recall=hit_accounts/target_accounts if target_accounts else np.nan,
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("results/payment_method_smurfing"))
    a=p.parse_args()

    print("Loading TRAIN warm-up + DEVELOPMENT...")
    train=load(a.train,"train"); dev=load(a.development,"development")
    start=dev["ts"].min()
    warm=train[train["ts"].ge(start-pd.Timedelta(days=120))]
    data=pd.concat([warm,dev],ignore_index=True)
    del train,warm

    rows=[]
    for days in [30,45,60,90]:
        print(f"Aggregating {days}D windows...")
        g=aggregate(data,start,days)
        for count in [3,5,8]:
            for med in [2500,3000,3500,4000]:
                for agg in [7500,10000,15000,20000]:
                    generic=(
                        g["tx_count"].ge(count)&g["median_amount"].lt(med)&
                        g["aggregate_amount"].ge(agg)&
                        g["unique_receivers"].le(3)&
                        g["cross_rate"].le(.15)&g["fx_rate"].le(.15))
                    cash=(
                        g["cash_tx_count"].ge(count)&
                        g["cash_median_amount"].lt(med)&
                        g["cash_aggregate_amount"].ge(agg)&
                        g["cash_unique_receivers"].le(3)&
                        g["cash_cross_rate"].le(.15)&g["cash_fx_rate"].le(.15))

                    r=score(g,generic,"")
                    r.update(Model="Generic",Window_days=days,Min_count=count,
                             Median_ceiling=med,Aggregate_floor=agg)
                    rows.append(r)
                    r=score(g,cash,"cash_")
                    r.update(Model="Cash_Deposit_Aware",Window_days=days,
                             Min_count=count,Median_ceiling=med,
                             Aggregate_floor=agg)
                    rows.append(r)

    res=pd.DataFrame(rows)
    # Show useful candidates rather than simply maximizing recall.
    eligible=res[res["Trigger_rate"].le(.01)].copy()
    top=eligible.sort_values(
        ["Smurfing_recall","Precision","Trigger_rate"],
        ascending=[False,False,True]).groupby("Model",group_keys=False).head(10)

    a.output_dir.mkdir(parents=True,exist_ok=True)
    res.to_csv(a.output_dir/"generic_vs_cash_deposit_grid.csv",index=False)
    top.to_csv(a.output_dir/"generic_vs_cash_deposit_shortlist.csv",index=False)

    print("\n=== GENERIC VS CASH-DEPOSIT-AWARE SMURFING ===")
    for model,g in top.groupby("Model"):
        print(f"\n[{model}]")
        print(g.to_string(index=False))
    print("\nAll metrics are DEVELOPMENT-only. HOLDOUT was not accessed.")


if __name__=="__main__":
    main()
