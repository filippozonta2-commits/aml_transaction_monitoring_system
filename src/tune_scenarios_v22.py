"""V2.2 tuning with historical warm-up.

Uses the tail of TRAIN only as pre-development history. All reported alerts,
labels and metrics belong exclusively to DEVELOPMENT. HOLDOUT is never read.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load_period(path, period):
    df = pd.read_csv(path)
    df["Transaction_timestamp"] = pd.to_datetime(
        pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        + " " + df["Time"].astype(str), errors="coerce"
    )
    df["Period"] = period
    df["Cross_border"] = (
        df["Sender_bank_location"] != df["Receiver_bank_location"]
    ).astype("int8")
    df["Currency_mismatch"] = (
        df["Payment_currency"] != df["Received_currency"]
    ).astype("int8")
    return df


def build_history(train, development, warmup_days=120):
    dev_start = development["Transaction_timestamp"].min()
    cutoff = dev_start - pd.Timedelta(days=warmup_days)
    warm = train.loc[train["Transaction_timestamp"].ge(cutoff)].copy()
    combined = pd.concat([warm, development], ignore_index=True)
    return combined.sort_values("Transaction_timestamp"), dev_start


def sender_calendar_features(df, window_days):
    """Aggregate sender behavior in calendar windows anchored to dev start."""
    out = df.copy()
    origin = out["Transaction_timestamp"].min().normalize()
    elapsed = (
        out["Transaction_timestamp"] - origin
    ).dt.total_seconds() / 86400
    out["Window_id"] = np.floor(elapsed / window_days).astype("int32")

    agg = out.groupby(["Window_id", "Sender_account"], observed=True).agg(
        tx_count=("Amount", "size"),
        aggregate_amount=("Amount", "sum"),
        median_amount=("Amount", "median"),
        unique_receivers=("Receiver_account", "nunique"),
        cross_border_rate=("Cross_border", "mean"),
        currency_mismatch_rate=("Currency_mismatch", "mean"),
        first_tx=("Transaction_timestamp", "min"),
        last_tx=("Transaction_timestamp", "max"),
    ).reset_index()
    return out, agg


def evaluate_flag(tx, accounts, mask, typology):
    flagged = accounts.loc[mask, ["Window_id", "Sender_account"]]
    marked = flagged.assign(_hit=1)
    eval_tx = tx.merge(
        marked, on=["Window_id", "Sender_account"], how="left"
    )
    eval_tx["_hit"] = eval_tx["_hit"].fillna(0).astype("int8")
    eval_tx = eval_tx[eval_tx["Period"].eq("development")]

    hit = eval_tx["_hit"].eq(1)
    aml = eval_tx["Is_laundering"].eq(1)
    target = aml & eval_tx["Laundering_type"].eq(typology)
    triggered = int(hit.sum())
    tp = int((hit & aml).sum())
    target_hits = int((hit & target).sum())

    flagged_dev_accounts = eval_tx.loc[hit, "Sender_account"].nunique()
    target_accounts = eval_tx.loc[target, "Sender_account"].nunique()
    target_hit_accounts = eval_tx.loc[hit & target, "Sender_account"].nunique()

    return {
        "Triggered_transactions": triggered,
        "Triggered_accounts": flagged_dev_accounts,
        "Trigger_rate": triggered / len(eval_tx),
        "AML_cases": tp,
        "Precision": tp / triggered if triggered else np.nan,
        f"{typology}_transactions": int(target.sum()),
        f"{typology}_hits": target_hits,
        f"{typology}_recall": target_hits / int(target.sum()) if target.sum() else np.nan,
        f"{typology}_accounts": target_accounts,
        f"{typology}_hit_accounts": target_hit_accounts,
        f"{typology}_account_recall":
            target_hit_accounts / target_accounts if target_accounts else np.nan,
    }


def tune_smurfing(history):
    rows = []
    for days in [60, 90, 120]:
        tx, accounts = sender_calendar_features(history, days)
        for min_count in [3, 5, 8, 10]:
            for median_ceiling in [3000, 4000, 5000, 6000]:
                for aggregate_floor in [10000, 20000, 30000]:
                    mask = (
                        accounts["tx_count"].ge(min_count)
                        & accounts["median_amount"].lt(median_ceiling)
                        & accounts["aggregate_amount"].ge(aggregate_floor)
                        & accounts["unique_receivers"].le(3)
                        & accounts["cross_border_rate"].le(.15)
                        & accounts["currency_mismatch_rate"].le(.15)
                    )
                    result = evaluate_flag(
                        tx, accounts, mask, "Smurfing"
                    )
                    result.update({
                        "Window_days": days,
                        "Min_count": min_count,
                        "Median_ceiling": median_ceiling,
                        "Aggregate_floor": aggregate_floor,
                    })
                    rows.append(result)
    return pd.DataFrame(rows)


def tune_fanout(history):
    rows = []
    for days in [7, 10, 14, 21]:
        tx, accounts = sender_calendar_features(history, days)
        for min_cp in [3, 4, 5, 6]:
            for max_tx in [12, 16, 20, 30]:
                for risk in [.10, .15, .20, .25, .30]:
                    mask = (
                        accounts["unique_receivers"].ge(min_cp)
                        & accounts["tx_count"].between(min_cp, max_tx)
                        & (
                            accounts["cross_border_rate"].ge(risk)
                            | accounts["currency_mismatch_rate"].ge(risk)
                        )
                    )
                    result = evaluate_flag(
                        tx, accounts, mask, "Fan_Out"
                    )
                    result.update({
                        "Window_days": days,
                        "Min_counterparties": min_cp,
                        "Max_transactions": max_tx,
                        "Geo_FX_threshold": risk,
                    })
                    rows.append(result)
    return pd.DataFrame(rows)


def shortlist(df, recall_col, max_trigger_rate=.01, n=12):
    eligible = df[df["Trigger_rate"].le(max_trigger_rate)].copy()
    if eligible.empty:
        eligible = df.copy()
    return eligible.sort_values(
        [recall_col, "Precision", "Trigger_rate"],
        ascending=[False, False, True],
    ).head(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path,
                   default=Path("data/temporal/SAML-D_train.csv"))
    p.add_argument("--development", type=Path,
                   default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/scenario_tuning_v22"))
    p.add_argument("--warmup-days", type=int, default=120)
    args = p.parse_args()

    print("Loading TRAIN history and DEVELOPMENT...")
    train = load_period(args.train, "train")
    dev = load_period(args.development, "development")
    history, dev_start = build_history(train, dev, args.warmup_days)
    print(f"Development starts: {dev_start}")
    print(f"Development rows: {len(dev):,}")
    print(f"Warm-up + development rows: {len(history):,}")
    print("HOLDOUT is not accessed.")

    print("\nTuning Smurfing with historical warm-up...")
    smurf = tune_smurfing(history)
    print("Tuning Fan-Out across intermediate configurations...")
    fan = tune_fanout(history)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    smurf.to_csv(args.output_dir / "smurfing_v22_grid.csv", index=False)
    fan.to_csv(args.output_dir / "fanout_v22_grid.csv", index=False)

    smurf_top = shortlist(smurf, "Smurfing_recall", .01)
    fan_top = shortlist(fan, "Fan_Out_recall", .02)

    print("\n=== V2.2 SMURFING SHORTLIST ===")
    print(smurf_top.to_string(index=False))
    print("\n=== V2.2 FAN-OUT SHORTLIST ===")
    print(fan_top.to_string(index=False))
    print("\nAll metrics above are DEVELOPMENT-only.")


if __name__ == "__main__":
    main()
