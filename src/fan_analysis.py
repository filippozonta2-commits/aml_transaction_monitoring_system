"""Compare normal and laundering fan-in/fan-out behavior in SAML-D.

This diagnostic script is intentionally descriptive: Laundering_type is used
only to understand ground-truth patterns and design later detection scenarios.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


TARGET_TYPES = {
    "Normal_Fan_In",
    "Normal_Small_Fan_In",
    "Normal_Fan_Out",
    "Normal_Small_Fan_Out",
    "Fan_In",
    "Fan_Out",
    "Layered_Fan_In",
    "Layered_Fan_Out",
}


def load_fan_transactions(path: Path, chunksize: int) -> pd.DataFrame:
    usecols = [
        "Time", "Date", "Sender_account", "Receiver_account", "Amount",
        "Payment_currency", "Received_currency", "Sender_bank_location",
        "Receiver_bank_location", "Payment_type", "Is_laundering",
        "Laundering_type",
    ]
    parts = []

    for n, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1
    ):
        selected = chunk[chunk["Laundering_type"].isin(TARGET_TYPES)].copy()
        if not selected.empty:
            parts.append(selected)
        print(f"Processed chunk {n:,} | fan-pattern rows retained: {len(selected):,}")

    if not parts:
        raise ValueError("No target fan typologies were found in the dataset.")

    data = pd.concat(parts, ignore_index=True)
    data["Timestamp"] = pd.to_datetime(
        pd.to_datetime(data["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        + " " + data["Time"].astype(str),
        errors="coerce",
    )
    data["Cross_border"] = (
        data["Sender_bank_location"] != data["Receiver_bank_location"]
    ).astype(int)
    data["Currency_mismatch"] = (
        data["Payment_currency"] != data["Received_currency"]
    ).astype(int)
    return data


def transaction_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for typology, g in data.groupby("Laundering_type"):
        rows.append({
            "Laundering_type": typology,
            "Transactions": len(g),
            "AML_rate": g["Is_laundering"].mean(),
            "Amount_median": g["Amount"].median(),
            "Amount_p95": g["Amount"].quantile(.95),
            "Amount_mean": g["Amount"].mean(),
            "Cross_border_rate": g["Cross_border"].mean(),
            "Currency_mismatch_rate": g["Currency_mismatch"].mean(),
            "Unique_senders": g["Sender_account"].nunique(),
            "Unique_receivers": g["Receiver_account"].nunique(),
        })
    return pd.DataFrame(rows).sort_values("Transactions", ascending=False)


def focal_account_summary(data: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for typology, g in data.groupby("Laundering_type"):
        direction = "in" if "Fan_In" in typology else "out"
        focal = "Receiver_account" if direction == "in" else "Sender_account"
        counterparty = "Sender_account" if direction == "in" else "Receiver_account"

        grouped = g.groupby(focal).agg(
            Transaction_count=("Amount", "size"),
            Aggregate_amount=("Amount", "sum"),
            Median_amount=("Amount", "median"),
            Unique_counterparties=(counterparty, "nunique"),
            First_timestamp=("Timestamp", "min"),
            Last_timestamp=("Timestamp", "max"),
            Cross_border_rate=("Cross_border", "mean"),
            Currency_mismatch_rate=("Currency_mismatch", "mean"),
        ).reset_index()

        grouped["Duration_hours"] = (
            grouped["Last_timestamp"] - grouped["First_timestamp"]
        ).dt.total_seconds() / 3600
        grouped["Laundering_type"] = typology
        grouped["Direction"] = direction
        frames.append(grouped)

    return pd.concat(frames, ignore_index=True)


def summarize_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Transaction_count", "Aggregate_amount", "Median_amount",
        "Unique_counterparties", "Duration_hours",
        "Cross_border_rate", "Currency_mismatch_rate",
    ]
    rows = []
    for typology, g in accounts.groupby("Laundering_type"):
        row = {"Laundering_type": typology, "Focal_accounts": len(g)}
        for metric in metrics:
            row[f"{metric}_median"] = g[metric].median()
            row[f"{metric}_p95"] = g[metric].quantile(.95)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Focal_accounts", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/SAML-D.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fan_analysis"))
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_fan_transactions(args.input, args.chunksize)
    tx = transaction_summary(data)
    accounts = focal_account_summary(data)
    account_summary = summarize_accounts(accounts)

    tx.to_csv(args.output_dir / "fan_transaction_summary.csv", index=False)
    accounts.to_csv(args.output_dir / "fan_focal_accounts.csv", index=False)
    account_summary.to_csv(args.output_dir / "fan_account_summary.csv", index=False)

    print("\n=== FAN TRANSACTION SUMMARY ===")
    print(tx.to_string(index=False))
    print("\n=== FAN FOCAL-ACCOUNT SUMMARY ===")
    print(account_summary.to_string(index=False))
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
