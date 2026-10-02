"""Compare normal sub-threshold behavior with AML Structuring and Smurfing.

Laundering_type is used only for diagnostic ground-truth analysis. It is not
used by the production scenario logic.
"""

from pathlib import Path
import argparse

import pandas as pd


TARGET_TYPES = {
    "Structuring",
    "Smurfing",
}

NORMAL_PREFIXES = (
    "Normal",
)


def load_relevant_transactions(path: Path, chunksize: int) -> pd.DataFrame:
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
        # Retain all AML Structuring/Smurfing rows plus normal transactions
        # below 10k as a comparison population.
        aml_mask = chunk["Laundering_type"].isin(TARGET_TYPES)
        normal_mask = (
            (chunk["Is_laundering"] == 0)
            & chunk["Amount"].lt(10_000)
        )
        selected = chunk.loc[aml_mask | normal_mask].copy()

        # Sampling the huge normal population keeps account-level diagnostics
        # tractable while AML rows are retained in full.
        normal = selected[selected["Is_laundering"] == 0]
        aml = selected[selected["Is_laundering"] == 1]
        if len(normal) > 25_000:
            normal = normal.sample(n=25_000, random_state=42 + n)

        parts.append(pd.concat([normal, aml], ignore_index=True))
        print(
            f"Processed chunk {n:,} | normal comparison: {len(normal):,} | "
            f"AML structuring/smurfing: {len(aml):,}"
        )

    data = pd.concat(parts, ignore_index=True)
    data["Pattern"] = data["Laundering_type"].where(
        data["Is_laundering"].eq(1), "Normal_Sub_10k"
    )
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
    for pattern, g in data.groupby("Pattern"):
        rows.append({
            "Pattern": pattern,
            "Transactions": len(g),
            "Amount_median": g["Amount"].median(),
            "Amount_p25": g["Amount"].quantile(.25),
            "Amount_p75": g["Amount"].quantile(.75),
            "Amount_p95": g["Amount"].quantile(.95),
            "Cross_border_rate": g["Cross_border"].mean(),
            "Currency_mismatch_rate": g["Currency_mismatch"].mean(),
            "Unique_senders": g["Sender_account"].nunique(),
            "Unique_receivers": g["Receiver_account"].nunique(),
        })
    return pd.DataFrame(rows)


def sender_summary(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for pattern, g in data.groupby("Pattern"):
        grouped = g.groupby("Sender_account").agg(
            Transaction_count=("Amount", "size"),
            Aggregate_amount=("Amount", "sum"),
            Median_amount=("Amount", "median"),
            Unique_receivers=("Receiver_account", "nunique"),
            First_timestamp=("Timestamp", "min"),
            Last_timestamp=("Timestamp", "max"),
            Cross_border_rate=("Cross_border", "mean"),
            Currency_mismatch_rate=("Currency_mismatch", "mean"),
        ).reset_index()

        grouped["Duration_hours"] = (
            grouped["Last_timestamp"] - grouped["First_timestamp"]
        ).dt.total_seconds() / 3600
        grouped["Pattern"] = pattern
        frames.append(grouped)

    return pd.concat(frames, ignore_index=True)


def summarize_senders(senders: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Transaction_count", "Aggregate_amount", "Median_amount",
        "Unique_receivers", "Duration_hours",
        "Cross_border_rate", "Currency_mismatch_rate",
    ]
    rows = []
    for pattern, g in senders.groupby("Pattern"):
        row = {"Pattern": pattern, "Sender_accounts": len(g)}
        for metric in metrics:
            row[f"{metric}_median"] = g[metric].median()
            row[f"{metric}_p95"] = g[metric].quantile(.95)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/SAML-D.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/structuring_analysis")
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_relevant_transactions(args.input, args.chunksize)
    tx = transaction_summary(data)
    senders = sender_summary(data)
    sender_stats = summarize_senders(senders)

    tx.to_csv(args.output_dir / "structuring_transaction_summary.csv", index=False)
    senders.to_csv(args.output_dir / "structuring_sender_accounts.csv", index=False)
    sender_stats.to_csv(args.output_dir / "structuring_sender_summary.csv", index=False)

    print("\n=== STRUCTURING TRANSACTION SUMMARY ===")
    print(tx.to_string(index=False))
    print("\n=== STRUCTURING SENDER SUMMARY ===")
    print(sender_stats.to_string(index=False))
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
