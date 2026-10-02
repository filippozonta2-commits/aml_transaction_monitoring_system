"""Fast V2.2 scenario tuning.

Pre-aggregates development evaluation statistics at sender/window level once.
Grid candidates are then evaluated entirely on compact account-window tables.
TRAIN is used only as historical warm-up; HOLDOUT is never read.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load_period(path, period):
    usecols = [
        "Time","Date","Sender_account","Receiver_account","Amount",
        "Sender_bank_location","Receiver_bank_location",
        "Payment_currency","Received_currency",
        "Is_laundering","Laundering_type",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["ts"] = pd.to_datetime(
        pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        + " " + df["Time"].astype(str), errors="coerce"
    )
    df["Period"] = period
    df["cross"] = (
        df["Sender_bank_location"] != df["Receiver_bank_location"]
    ).astype("int8")
    df["fx"] = (
        df["Payment_currency"] != df["Received_currency"]
    ).astype("int8")
    return df


def history_frame(train, dev, warmup_days):
    start = dev["ts"].min()
    warm = train[train["ts"].ge(start - pd.Timedelta(days=warmup_days))]
    return pd.concat([warm, dev], ignore_index=True), start


def aggregate_sender_windows(data, dev_start, days, target):
    # Windows are anchored at development start, so preceding negative window
    # IDs provide warm-up context without turning train rows into evaluation.
    elapsed = (data["ts"] - dev_start).dt.total_seconds() / 86400
    x = data.copy()
    x["window_id"] = np.floor(elapsed / days).astype("int32")
    x["dev_row"] = x["Period"].eq("development").astype("int8")
    x["dev_aml"] = (
        x["Period"].eq("development") & x["Is_laundering"].eq(1)
    ).astype("int8")
    x["dev_target"] = (
        x["Period"].eq("development")
        & x["Is_laundering"].eq(1)
        & x["Laundering_type"].eq(target)
    ).astype("int8")

    g = x.groupby(["window_id","Sender_account"], observed=True).agg(
        tx_count=("Amount","size"),
        aggregate_amount=("Amount","sum"),
        median_amount=("Amount","median"),
        unique_receivers=("Receiver_account","nunique"),
        cross_border_rate=("cross","mean"),
        currency_mismatch_rate=("fx","mean"),
        dev_transactions=("dev_row","sum"),
        dev_aml=("dev_aml","sum"),
        dev_target=("dev_target","sum"),
    ).reset_index()

    # Only groups containing development rows can generate evaluated alerts.
    return g[g["dev_transactions"].gt(0)].copy()


def metrics(g, mask, target):
    h = g.loc[mask]
    triggered = int(h["dev_transactions"].sum())
    aml_hits = int(h["dev_aml"].sum())
    target_hits = int(h["dev_target"].sum())
    total_dev = int(g["dev_transactions"].sum())
    total_target = int(g["dev_target"].sum())

    # Account-level target recall.
    target_accounts = g.loc[g["dev_target"].gt(0), "Sender_account"].nunique()
    hit_accounts = h.loc[h["dev_target"].gt(0), "Sender_account"].nunique()

    return {
        "Triggered_transactions": triggered,
        "Triggered_account_windows": len(h),
        "Trigger_rate": triggered / total_dev if total_dev else np.nan,
        "AML_cases": aml_hits,
        "Precision": aml_hits / triggered if triggered else np.nan,
        f"{target}_transactions": total_target,
        f"{target}_hits": target_hits,
        f"{target}_recall": target_hits / total_target if total_target else np.nan,
        f"{target}_accounts": target_accounts,
        f"{target}_hit_accounts": hit_accounts,
        f"{target}_account_recall":
            hit_accounts / target_accounts if target_accounts else np.nan,
    }


def tune_smurf(data, dev_start):
    rows = []
    for days in [60,90,120]:
        print(f"  Smurfing: aggregating {days}D windows...")
        g = aggregate_sender_windows(data, dev_start, days, "Smurfing")
        for count in [3,5,8,10]:
            for med in [3000,4000,5000,6000]:
                for agg in [10000,20000,30000]:
                    mask = (
                        g["tx_count"].ge(count)
                        & g["median_amount"].lt(med)
                        & g["aggregate_amount"].ge(agg)
                        & g["unique_receivers"].le(3)
                        & g["cross_border_rate"].le(.15)
                        & g["currency_mismatch_rate"].le(.15)
                    )
                    r = metrics(g, mask, "Smurfing")
                    r.update(Window_days=days, Min_count=count,
                             Median_ceiling=med, Aggregate_floor=agg)
                    rows.append(r)
    return pd.DataFrame(rows)


def tune_fan(data, dev_start):
    rows = []
    for days in [7,10,14,21]:
        print(f"  Fan-Out: aggregating {days}D windows...")
        g = aggregate_sender_windows(data, dev_start, days, "Fan_Out")
        for cp in [3,4,5,6]:
            for maxtx in [12,16,20,30]:
                for risk in [.10,.15,.20,.25,.30]:
                    mask = (
                        g["unique_receivers"].ge(cp)
                        & g["tx_count"].between(cp, maxtx)
                        & (
                            g["cross_border_rate"].ge(risk)
                            | g["currency_mismatch_rate"].ge(risk)
                        )
                    )
                    r = metrics(g, mask, "Fan_Out")
                    r.update(Window_days=days, Min_counterparties=cp,
                             Max_transactions=maxtx, Geo_FX_threshold=risk)
                    rows.append(r)
    return pd.DataFrame(rows)


def shortlist(df, recall, limit, n=12):
    x = df[df["Trigger_rate"].le(limit)]
    if x.empty:
        x = df
    return x.sort_values(
        [recall,"Precision","Trigger_rate"],
        ascending=[False,False,True]
    ).head(n)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("results/scenario_tuning_v22_fast"))
    p.add_argument("--warmup-days",type=int,default=120)
    a=p.parse_args()

    print("Loading TRAIN + DEVELOPMENT...")
    train=load_period(a.train,"train")
    dev=load_period(a.development,"development")
    data,start=history_frame(train,dev,a.warmup_days)
    del train
    print(f"Development rows: {len(dev):,}")
    print(f"Warm-up + development rows: {len(data):,}")
    print("HOLDOUT is not accessed.")

    print("\nFast Smurfing tuning...")
    sm=tune_smurf(data,start)
    print("Fast Fan-Out tuning...")
    fo=tune_fan(data,start)

    a.output_dir.mkdir(parents=True,exist_ok=True)
    sm.to_csv(a.output_dir/"smurfing_fast_grid.csv",index=False)
    fo.to_csv(a.output_dir/"fanout_fast_grid.csv",index=False)

    print("\n=== V2.2 FAST SMURFING SHORTLIST ===")
    print(shortlist(sm,"Smurfing_recall",.01).to_string(index=False))
    print("\n=== V2.2 FAST FAN-OUT SHORTLIST ===")
    print(shortlist(fo,"Fan_Out_recall",.02).to_string(index=False))
    print("\nAll reported metrics are DEVELOPMENT-only.")


if __name__=="__main__":
    main()
