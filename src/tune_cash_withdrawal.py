"""Tune a channel-aware Cash Withdrawal scenario on DEVELOPMENT.

TRAIN provides historical warm-up only. HOLDOUT is never accessed.
The detector uses account/window behavior within Payment_type == Cash Withdrawal,
rather than treating the payment channel itself as an alert.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load(path, period):
    cols=["Time","Date","Sender_account","Receiver_account","Amount",
          "Payment_type","Is_laundering","Laundering_type"]
    d=pd.read_csv(path,usecols=cols)
    d["ts"]=pd.to_datetime(
        pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")
        +" "+d["Time"].astype(str),errors="coerce")
    d["Period"]=period
    return d


def aggregate(data,start,days):
    x=data[data["Payment_type"].eq("Cash Withdrawal")].copy()
    x["window_id"]=np.floor(
        (x["ts"]-start).dt.total_seconds()/86400/days
    ).astype("int32")
    x["dev"]=x["Period"].eq("development").astype("int8")
    x["dev_aml"]=(x["Period"].eq("development")&
                  x["Is_laundering"].eq(1)).astype("int8")
    x["dev_target"]=(x["Period"].eq("development")&
                     x["Is_laundering"].eq(1)&
                     x["Laundering_type"].eq("Cash_Withdrawal")).astype("int8")

    g=x.groupby(["window_id","Sender_account"],observed=True).agg(
        tx_count=("Amount","size"),
        aggregate_amount=("Amount","sum"),
        median_amount=("Amount","median"),
        max_amount=("Amount","max"),
        amount_std=("Amount","std"),
        dev_transactions=("dev","sum"),
        dev_aml=("dev_aml","sum"),
        dev_target=("dev_target","sum"),
    ).reset_index()
    return g[g["dev_transactions"].gt(0)].copy()


def metrics(g,mask):
    h=g.loc[mask]
    trig=int(h["dev_transactions"].sum())
    aml=int(h["dev_aml"].sum())
    hits=int(h["dev_target"].sum())
    total_tx=int(g["dev_transactions"].sum())
    total_target=int(g["dev_target"].sum())
    target_accounts=g.loc[g["dev_target"].gt(0),"Sender_account"].nunique()
    hit_accounts=h.loc[h["dev_target"].gt(0),"Sender_account"].nunique()
    return dict(
        Triggered_transactions=trig,
        Trigger_rate=trig/total_tx if total_tx else np.nan,
        AML_cases=aml,
        Precision=aml/trig if trig else np.nan,
        Cash_Withdrawal_hits=hits,
        Cash_Withdrawal_recall=hits/total_target if total_target else np.nan,
        Cash_Withdrawal_accounts=target_accounts,
        Cash_Withdrawal_hit_accounts=hit_accounts,
        Cash_Withdrawal_account_recall=hit_accounts/target_accounts if target_accounts else np.nan,
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("results/cash_withdrawal_tuning"))
    a=p.parse_args()

    print("Loading TRAIN warm-up + DEVELOPMENT...")
    train=load(a.train,"train"); dev=load(a.development,"development")
    start=dev["ts"].min()
    warm=train[train["ts"].ge(start-pd.Timedelta(days=60))]
    data=pd.concat([warm,dev],ignore_index=True)
    del train,warm
    print(f"Development cash-withdrawal transactions: {(dev['Payment_type']=='Cash Withdrawal').sum():,}")
    print(f"Development target AML transactions: {((dev['Is_laundering']==1)&(dev['Laundering_type']=='Cash_Withdrawal')).sum():,}")
    print("HOLDOUT is not accessed.")

    rows=[]
    for days in [3,7,14,30]:
        print(f"Aggregating {days}D Cash Withdrawal windows...")
        g=aggregate(data,start,days)
        for count in [2,3,4,5,8]:
            for agg in [300,500,750,1000,1500]:
                for medceil in [150,200,250,300]:
                    mask=(g["tx_count"].ge(count)&
                          g["aggregate_amount"].ge(agg)&
                          g["median_amount"].le(medceil))
                    r=metrics(g,mask)
                    r.update(Window_days=days,Min_count=count,
                             Aggregate_floor=agg,Median_ceiling=medceil)
                    rows.append(r)

    res=pd.DataFrame(rows)
    eligible=res[res["Trigger_rate"].le(.10)].copy()
    top=eligible.sort_values(
        ["Cash_Withdrawal_recall","Precision","Trigger_rate"],
        ascending=[False,False,True]).head(15)

    a.output_dir.mkdir(parents=True,exist_ok=True)
    res.to_csv(a.output_dir/"cash_withdrawal_grid.csv",index=False)
    top.to_csv(a.output_dir/"cash_withdrawal_shortlist.csv",index=False)

    print("\n=== CASH WITHDRAWAL SCENARIO SHORTLIST ===")
    print(top.to_string(index=False))
    print("\nTrigger_rate is relative to Cash Withdrawal transactions, not the full dataset.")
    print("All metrics are DEVELOPMENT-only. HOLDOUT was not accessed.")


if __name__=="__main__":
    main()
