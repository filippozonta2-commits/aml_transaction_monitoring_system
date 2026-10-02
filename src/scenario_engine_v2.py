"""Scenario Engine V2: temporal-development AML detectors.

Initial V2 focuses on four patterns identified during diagnostic EDA:
Fan-In, Fan-Out, Structuring, and Smurfing.

Ground-truth labels are never used as detector inputs.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


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
    return out.sort_values("Transaction_timestamp", kind="stable")


def window_features(df, freq="10D"):
    """Vectorized focal-account features using fixed time buckets.

    Fixed buckets are used for scalable V2 development. A later production
    layer can replace them with exact sliding windows after scenario tuning.
    """
    out = df.copy()
    out["Window"] = out["Transaction_timestamp"].dt.floor(freq)

    sender_keys = ["Window", "Sender_account"]
    receiver_keys = ["Window", "Receiver_account"]

    sender = out.groupby(sender_keys, observed=True).agg(
        Sender_tx_count=("Amount", "size"),
        Sender_aggregate_amount=("Amount", "sum"),
        Sender_unique_receivers=("Receiver_account", "nunique"),
        Sender_cross_border_rate=("Cross_border", "mean"),
        Sender_currency_mismatch_rate=("Currency_mismatch", "mean"),
        Sender_median_amount=("Amount", "median"),
    ).reset_index()

    receiver = out.groupby(receiver_keys, observed=True).agg(
        Receiver_tx_count=("Amount", "size"),
        Receiver_aggregate_amount=("Amount", "sum"),
        Receiver_unique_senders=("Sender_account", "nunique"),
        Receiver_cross_border_rate=("Cross_border", "mean"),
        Receiver_currency_mismatch_rate=("Currency_mismatch", "mean"),
        Receiver_median_amount=("Amount", "median"),
    ).reset_index()

    out = out.merge(sender, on=sender_keys, how="left")
    out = out.merge(receiver, on=receiver_keys, how="left")
    return out


def apply_v2(df):
    out = df.copy()

    # Smurfing: repeated relatively small outgoing transactions by one sender.
    out["SCN_SMURFING"] = (
        out["Sender_tx_count"].ge(8)
        & out["Sender_median_amount"].lt(5_000)
        & out["Sender_aggregate_amount"].ge(20_000)
        & out["Sender_unique_receivers"].le(3)
    ).astype("int8")

    # Structuring: sub-threshold transfers concentrating into a receiver from
    # multiple senders, with additional cross-border / FX risk signal.
    out["SCN_STRUCTURING_V2"] = (
        out["Amount"].lt(10_000)
        & out["Receiver_unique_senders"].ge(5)
        & out["Receiver_aggregate_amount"].ge(20_000)
        & (
            out["Receiver_cross_border_rate"].ge(.20)
            | out["Receiver_currency_mismatch_rate"].ge(.25)
        )
    ).astype("int8")

    # Fan-Out: moderate distinct-counterparty dispersion over a longer window,
    # augmented by geography/FX risk to distinguish common normal fan behavior.
    out["SCN_FAN_OUT_V2"] = (
        out["Sender_unique_receivers"].between(5, 12)
        & out["Sender_tx_count"].between(5, 20)
        & (
            out["Sender_cross_border_rate"].ge(.20)
            | out["Sender_currency_mismatch_rate"].ge(.25)
        )
    ).astype("int8")

    # Fan-In: analogous concentration on the receiver side.
    out["SCN_FAN_IN_V2"] = (
        out["Receiver_unique_senders"].between(5, 15)
        & out["Receiver_tx_count"].between(5, 20)
        & (
            out["Receiver_cross_border_rate"].ge(.10)
            | out["Receiver_currency_mismatch_rate"].ge(.20)
        )
    ).astype("int8")

    flags = [
        "SCN_SMURFING", "SCN_STRUCTURING_V2",
        "SCN_FAN_OUT_V2", "SCN_FAN_IN_V2",
    ]
    out["Scenario_V2_alert"] = out[flags].max(axis=1)
    return out, flags


def evaluate(df, flags):
    rows = []
    for flag in flags:
        hit = df[flag].eq(1)
        aml = df["Is_laundering"].eq(1)
        tp = int((hit & aml).sum())
        triggered = int(hit.sum())
        rows.append({
            "Scenario": flag,
            "Triggered": triggered,
            "Trigger_rate": triggered / len(df),
            "AML_cases": tp,
            "Precision": tp / triggered if triggered else np.nan,
            "AML_recall": tp / int(aml.sum()) if aml.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def typology_coverage(df, flags):
    aml = df[df["Is_laundering"].eq(1)]
    rows = []
    for typology, g in aml.groupby("Laundering_type"):
        row = {
            "Laundering_type": typology,
            "AML_transactions": len(g),
        }
        for flag in flags:
            row[flag] = g[flag].mean()
        row["Any_V2_recall"] = g["Scenario_V2_alert"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("AML_transactions", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/temporal/SAML-D_development.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/scenario_v2"),
    )
    parser.add_argument("--window", default="10D")
    args = parser.parse_args()

    print("Loading temporal development set...")
    df = pd.read_csv(args.input)
    print(f"Rows: {len(df):,} | AML: {int(df['Is_laundering'].sum()):,}")

    print(f"Building scalable focal-account features ({args.window} buckets)...")
    df = prepare(df)
    df = window_features(df, args.window)

    print("Applying Scenario Engine V2...")
    df, flags = apply_v2(df)
    summary = evaluate(df, flags)
    coverage = typology_coverage(df, flags)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "scenario_v2_summary.csv", index=False)
    coverage.to_csv(args.output_dir / "scenario_v2_typology_coverage.csv", index=False)

    alerts = df[df["Scenario_V2_alert"].eq(1)].copy()
    keep = [
        "Transaction_timestamp", "Sender_account", "Receiver_account", "Amount",
        "Is_laundering", "Laundering_type",
    ] + flags
    alerts[keep].to_csv(args.output_dir / "scenario_v2_alerts.csv", index=False)

    total_alerts = int(df["Scenario_V2_alert"].sum())
    aml = df["Is_laundering"].eq(1)
    aml_hits = int((df["Scenario_V2_alert"].eq(1) & aml).sum())

    print("\n=== SCENARIO ENGINE V2 ===")
    print(f"Transactions evaluated: {len(df):,}")
    print(f"Alerts generated:       {total_alerts:,}")
    print(f"Alert rate:             {total_alerts / len(df):.4%}")
    print(f"AML transactions:       {int(aml.sum()):,}")
    print(f"AML transactions hit:   {aml_hits:,}")
    print(f"Overall AML recall:     {aml_hits / int(aml.sum()):.2%}")
    print("\nScenario summary:")
    print(summary.to_string(index=False))

    focus = coverage[
        coverage["Laundering_type"].isin(
            ["Smurfing", "Structuring", "Fan_In", "Fan_Out"]
        )
    ]
    print("\nTarget typology coverage:")
    print(focus.to_string(index=False))
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
