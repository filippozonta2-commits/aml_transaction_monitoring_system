"""Four-scenario causal AML engine V3: DEVELOPMENT-only baseline.

Receiver thresholds are initial V2-inspired rules, not yet tuned.
"""
from pathlib import Path
import pandas as pd
from src.tune_scenarios_v22 import load_period, build_history
from src.scenario_features_v3 import sender_rolling_features
from src.receiver_features_v3 import receiver_rolling_features


def metrics(labels, flags):
    alerts = int(flags.sum())
    hits = int((flags & labels).sum())
    return {
        "Alerts": alerts, "AML_hits": hits,
        "Alert_rate": alerts / len(labels),
        "Precision": hits / alerts if alerts else 0.,
        "Overall_AML_recall": hits / int(labels.sum()) if labels.sum() else 0.,
    }


def main():
    train = load_period("data/temporal/SAML-D_train.csv", "train")
    dev = load_period("data/temporal/SAML-D_development.csv", "development")
    history, start = build_history(train, dev, 120)
    history = history.sort_values("Transaction_timestamp", kind="stable")
    if train["Transaction_timestamp"].max() >= start:
        raise ValueError("TRAIN must end before DEVELOPMENT")
    idx = history.index[history["Period"].eq("development")]
    labels = history.loc[idx, "Is_laundering"].eq(1)
    print("Building 90-day sender features...", flush=True)
    sender90 = sender_rolling_features(history, 90)
    smurf = (
        sender90["tx_count"].ge(3) & sender90["median_amount"].lt(3000)
        & sender90["aggregate_amount"].ge(20000)
        & sender90["unique_receivers"].le(3)
        & sender90["cross_border_rate"].le(.15)
        & sender90["currency_mismatch_rate"].le(.15)
    ).loc[idx]
    del sender90
    print("Building 21-day sender features...", flush=True)
    sender21 = sender_rolling_features(history, 21)
    fanout = (
        sender21["unique_receivers"].ge(3)
        & sender21["tx_count"].between(3, 12)
        & (sender21["cross_border_rate"].ge(.30)
           | sender21["currency_mismatch_rate"].ge(.30))
    ).loc[idx]
    del sender21
    print("Building 10-day receiver features...", flush=True)
    receiver = receiver_rolling_features(history, 10)
    structuring = (
        receiver["Amount"].lt(10000)
        & receiver["receiver_unique_senders"].ge(5)
        & receiver["receiver_aggregate_amount"].ge(20000)
        & (receiver["receiver_cross_border_rate"].ge(.20)
           | receiver["receiver_currency_mismatch_rate"].ge(.25))
    ).loc[idx]
    fanin = (
        receiver["receiver_unique_senders"].between(5, 15)
        & receiver["receiver_tx_count"].between(5, 20)
        & (receiver["receiver_cross_border_rate"].ge(.10)
           | receiver["receiver_currency_mismatch_rate"].ge(.20))
    ).loc[idx]
    del receiver
    flags = {"Smurfing": smurf, "Fan_Out": fanout,
             "Structuring": structuring, "Fan_In": fanin}
    rows = []
    combined = pd.Series(False, index=idx)
    for name, flag in flags.items():
        combined = combined | flag
        rows.append({"Scenario": name, **metrics(labels, flag)})
    rows.append({"Scenario": "ALL_FOUR_DEDUPED", **metrics(labels, combined)})
    output_dir = Path("results/scenario_v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "development_summary.csv", index=False)
    print("\n=== V3 DEVELOPMENT SUMMARY ===", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"AML transactions: {int(labels.sum())}; total DEVELOPMENT: {len(idx)}")
    print("Saved results/scenario_v3/development_summary.csv; HOLDOUT untouched.")


if __name__ == "__main__":
    main()
