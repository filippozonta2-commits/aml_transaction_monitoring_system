"""Development-only freeze evaluation for Structuring and Fan-In.

Purpose
-------
Evaluate the existing V2 definitions for Structuring and Fan-In on the
chronological DEVELOPMENT split before they are promoted into the frozen
scenario configuration. No HOLDOUT file is read.

This is intentionally a validation/freeze step, not another broad tuning grid.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

USECOLS = [
    "Time","Date","Sender_account","Receiver_account","Amount",
    "Payment_currency","Received_currency",
    "Sender_bank_location","Receiver_bank_location",
    "Is_laundering","Laundering_type",
]

def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=USECOLS)
    d["ts"] = pd.to_datetime(
        pd.to_datetime(d["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        + " " + d["Time"].astype(str),
        errors="coerce",
    )
    d["cross"] = (
        d["Sender_bank_location"] != d["Receiver_bank_location"]
    ).astype("int8")
    d["fx"] = (
        d["Payment_currency"] != d["Received_currency"]
    ).astype("int8")
    return d.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--development",
        type=Path,
        default=Path("data/temporal/SAML-D_development.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/freeze_structuring_fanin"),
    )
    p.add_argument("--window", default="10D")
    a = p.parse_args()

    print("Loading DEVELOPMENT only...")
    d = load(a.development)
    print(f"Rows: {len(d):,} | AML: {int(d['Is_laundering'].sum()):,}")
    print("HOLDOUT is not accessed.")

    d["window"] = d["ts"].dt.floor(a.window)
    keys = ["window", "Receiver_account"]
    g = d.groupby(keys, observed=True).agg(
        receiver_tx_count=("Amount", "size"),
        receiver_aggregate_amount=("Amount", "sum"),
        receiver_unique_senders=("Sender_account", "nunique"),
        receiver_cross_border_rate=("cross", "mean"),
        receiver_currency_mismatch_rate=("fx", "mean"),
    ).reset_index()
    x = d.merge(g, on=keys, how="left")

    x["SCN_STRUCTURING_FREEZE"] = (
        x["Amount"].lt(10_000)
        & x["receiver_unique_senders"].ge(5)
        & x["receiver_aggregate_amount"].ge(20_000)
        & (
            x["receiver_cross_border_rate"].ge(.20)
            | x["receiver_currency_mismatch_rate"].ge(.25)
        )
    )

    x["SCN_FAN_IN_FREEZE"] = (
        x["receiver_unique_senders"].between(5, 15)
        & x["receiver_tx_count"].between(5, 20)
        & (
            x["receiver_cross_border_rate"].ge(.10)
            | x["receiver_currency_mismatch_rate"].ge(.20)
        )
    )

    aml = x["Is_laundering"].eq(1)
    rows = []
    for flag, target in [
        ("SCN_STRUCTURING_FREEZE", "Structuring"),
        ("SCN_FAN_IN_FREEZE", "Fan_In"),
    ]:
        hit = x[flag]
        trig = int(hit.sum())
        aml_hits = int((hit & aml).sum())
        target_mask = aml & x["Laundering_type"].eq(target)
        target_n = int(target_mask.sum())
        target_hits = int((hit & target_mask).sum())
        rows.append({
            "Scenario": flag,
            "Target_typology": target,
            "Triggered_transactions": trig,
            "Trigger_rate": trig / len(x),
            "AML_cases": aml_hits,
            "Precision": aml_hits / trig if trig else np.nan,
            "Target_transactions": target_n,
            "Target_hits": target_hits,
            "Target_recall": target_hits / target_n if target_n else np.nan,
        })

    out = pd.DataFrame(rows)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output_dir / "freeze_summary.csv", index=False)

    print("\n=== STRUCTURING + FAN-IN FREEZE CHECK ===")
    print(out.to_string(index=False))
    print(f"\nSaved: {a.output_dir/'freeze_summary.csv'}")
    print("Development-only. HOLDOUT was not accessed.")

if __name__ == "__main__":
    main()
