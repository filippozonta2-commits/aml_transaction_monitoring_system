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


def sender_calendar_features(df, window_days, dev_start):
    """Causal sender features within fixed, development-anchored buckets.

    Each row uses only transactions observed up to and including its timestamp.
    Equal-timestamp transactions use stable input order as a tie-breaker.
    Windows are fixed buckets, not sliding windows.
    """
    out = df.sort_values("Transaction_timestamp", kind="stable").copy()
    origin = pd.Timestamp(dev_start).normalize()
    elapsed = (out["Transaction_timestamp"] - origin).dt.total_seconds() / 86400
    out["Window_id"] = np.floor(elapsed / window_days).astype("int32")
    keys = ["Window_id", "Sender_account"]
    grouped = out.groupby(keys, sort=False, observed=True)
    out["tx_count"] = grouped.cumcount() + 1
    out["aggregate_amount"] = grouped["Amount"].cumsum()
    # Expanding median is causal; unlike a full-bucket median it cannot peek
    # at transactions that occur later in the window.
    out["median_amount"] = (
        grouped["Amount"].expanding().median().reset_index(level=keys, drop=True)
    )
    out["unique_receivers"] = (
        (~out.duplicated(keys + ["Receiver_account"])).astype("int32")
        .groupby([out[k] for k in keys], sort=False).cumsum()
    )
    out["cross_border_rate"] = grouped["Cross_border"].cumsum() / out["tx_count"]
    out["currency_mismatch_rate"] = (
        grouped["Currency_mismatch"].cumsum() / out["tx_count"]
    )
    return out


def evaluate_flag(tx, mask, typology):
    """Evaluate only development transactions; historical rows are context."""
    eval_tx = tx.loc[tx["Period"].eq("development")].copy()
    hit = mask.loc[eval_tx.index].astype(bool)
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


def tune_smurfing(history, dev_start):
    rows = []
    for days in [60, 90, 120]:
        tx = sender_calendar_features(history, days, dev_start)
        for min_count in [3, 5, 8, 10]:
            for median_ceiling in [3000, 4000, 5000, 6000]:
                for aggregate_floor in [10000, 20000, 30000]:
                    mask = (
                        tx["tx_count"].ge(min_count)
                        & tx["median_amount"].lt(median_ceiling)
                        & tx["aggregate_amount"].ge(aggregate_floor)
                        & tx["unique_receivers"].le(3)
                        & tx["cross_border_rate"].le(.15)
                        & tx["currency_mismatch_rate"].le(.15)
                    )
                    result = evaluate_flag(
                        tx, mask, "Smurfing"
                    )
                    result.update({
                        "Window_days": days,
                        "Min_count": min_count,
                        "Median_ceiling": median_ceiling,
                        "Aggregate_floor": aggregate_floor,
                    })
                    rows.append(result)
    return pd.DataFrame(rows)


def tune_fanout(history, dev_start):
    rows = []
    for days in [7, 10, 14, 21]:
        tx = sender_calendar_features(history, days, dev_start)
        for min_cp in [3, 4, 5, 6]:
            for max_tx in [12, 16, 20, 30]:
                for risk in [.10, .15, .20, .25, .30]:
                    mask = (
                        tx["unique_receivers"].ge(min_cp)
                        & tx["tx_count"].between(min_cp, max_tx)
                        & (
                            tx["cross_border_rate"].ge(risk)
                            | tx["currency_mismatch_rate"].ge(risk)
                        )
                    )
                    result = evaluate_flag(
                        tx, mask, "Fan_Out"
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
    smurf = tune_smurfing(history, dev_start)
    print("Tuning Fan-Out across intermediate configurations...")
    fan = tune_fanout(history, dev_start)

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
