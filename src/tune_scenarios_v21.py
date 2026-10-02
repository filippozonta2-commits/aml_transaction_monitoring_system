"""Scenario tuning V2.1 for Smurfing and Fan-Out on development data.

This script uses only the temporal development period. Ground-truth labels are
used for evaluation, never as detector inputs.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def prepare(df):
    out = df.copy()
    out["Transaction_timestamp"] = pd.to_datetime(
        pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        + " " + out["Time"].astype(str),
        errors="coerce",
    )
    out["Cross_border"] = (
        out["Sender_bank_location"] != out["Receiver_bank_location"]
    ).astype("int8")
    out["Currency_mismatch"] = (
        out["Payment_currency"] != out["Received_currency"]
    ).astype("int8")
    return out


def sender_period_features(df, freq):
    temp = df.copy()
    temp["Window"] = temp["Transaction_timestamp"].dt.to_period(freq).dt.start_time

    agg = temp.groupby(["Window", "Sender_account"], observed=True).agg(
        tx_count=("Amount", "size"),
        aggregate_amount=("Amount", "sum"),
        median_amount=("Amount", "median"),
        p95_amount=("Amount", lambda x: x.quantile(.95)),
        unique_receivers=("Receiver_account", "nunique"),
        cross_border_rate=("Cross_border", "mean"),
        currency_mismatch_rate=("Currency_mismatch", "mean"),
        first_tx=("Transaction_timestamp", "min"),
        last_tx=("Transaction_timestamp", "max"),
    ).reset_index()
    agg["duration_days"] = (
        agg["last_tx"] - agg["first_tx"]
    ).dt.total_seconds() / 86400
    return agg


def evaluate_account_detector(accounts, source, flag, typology):
    flagged = accounts[accounts[flag].eq(1)]
    flagged_keys = set(zip(flagged["Window"], flagged["Sender_account"]))

    temp = source.copy()
    temp["Window"] = temp["Transaction_timestamp"].dt.to_period(
        accounts.attrs["freq"]
    ).dt.start_time
    hit = pd.Series(
        [
            (w, s) in flagged_keys
            for w, s in zip(temp["Window"], temp["Sender_account"])
        ],
        index=temp.index,
    )

    aml = temp["Is_laundering"].eq(1)
    target = aml & temp["Laundering_type"].eq(typology)
    triggered = int(hit.sum())
    tp = int((hit & aml).sum())
    target_hits = int((hit & target).sum())

    return {
        "Detector": flag,
        "Triggered_transactions": triggered,
        "Trigger_rate": triggered / len(temp),
        "AML_cases": tp,
        "Precision": tp / triggered if triggered else np.nan,
        f"{typology}_cases": int(target.sum()),
        f"{typology}_hits": target_hits,
        f"{typology}_recall": target_hits / int(target.sum()) if target.sum() else np.nan,
    }


def smurfing_grid(df):
    """Test interpretable long-horizon parameter combinations."""
    accounts = sender_period_features(df, "60D")
    accounts.attrs["freq"] = "60D"
    rows = []

    for min_count in [5, 8, 10, 12]:
        for median_ceiling in [3000, 4000, 5000]:
            for aggregate_floor in [15000, 25000, 40000]:
                flag = (
                    accounts["tx_count"].ge(min_count)
                    & accounts["median_amount"].lt(median_ceiling)
                    & accounts["aggregate_amount"].ge(aggregate_floor)
                    & accounts["unique_receivers"].le(3)
                    & accounts["cross_border_rate"].le(.10)
                    & accounts["currency_mismatch_rate"].le(.10)
                )
                accounts["candidate"] = flag.astype("int8")
                result = evaluate_account_detector(
                    accounts, df, "candidate", "Smurfing"
                )
                result.update({
                    "Window": "60D",
                    "Min_count": min_count,
                    "Median_ceiling": median_ceiling,
                    "Aggregate_floor": aggregate_floor,
                })
                rows.append(result)

    return pd.DataFrame(rows).sort_values(
        ["Smurfing_recall", "Precision"], ascending=False
    )


def fanout_grid(df):
    accounts = sender_period_features(df, "10D")
    accounts.attrs["freq"] = "10D"
    rows = []

    for min_cp in [5, 6, 7, 8]:
        for max_cp in [10, 12]:
            for min_geo in [.20, .30, .40]:
                flag = (
                    accounts["unique_receivers"].between(min_cp, max_cp)
                    & accounts["tx_count"].between(min_cp, 20)
                    & (
                        accounts["cross_border_rate"].ge(min_geo)
                        | accounts["currency_mismatch_rate"].ge(min_geo)
                    )
                )
                accounts["candidate"] = flag.astype("int8")
                result = evaluate_account_detector(
                    accounts, df, "candidate", "Fan_Out"
                )
                result.update({
                    "Window": "10D",
                    "Min_counterparties": min_cp,
                    "Max_counterparties": max_cp,
                    "Geo_FX_threshold": min_geo,
                })
                rows.append(result)

    return pd.DataFrame(rows).sort_values(
        ["Fan_Out_recall", "Precision"], ascending=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/temporal/SAML-D_development.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/scenario_tuning_v21")
    )
    args = parser.parse_args()

    print("Loading development set only...")
    df = prepare(pd.read_csv(args.input))
    print(f"Rows: {len(df):,} | AML: {int(df['Is_laundering'].sum()):,}")

    print("Tuning long-horizon Smurfing detector...")
    smurf = smurfing_grid(df)

    print("Tuning selective Fan-Out detector...")
    fanout = fanout_grid(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    smurf.to_csv(args.output_dir / "smurfing_grid.csv", index=False)
    fanout.to_csv(args.output_dir / "fanout_grid.csv", index=False)

    print("\n=== TOP SMURFING CANDIDATES ===")
    print(smurf.head(10).to_string(index=False))

    print("\n=== TOP FAN-OUT CANDIDATES ===")
    print(fanout.head(10).to_string(index=False))

    print(
        "\nThese are development-set candidates only; no holdout data "
        "was accessed."
    )


if __name__ == "__main__":
    main()
