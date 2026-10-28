"""V3 rolling-window benchmark for Smurfing and Fan-Out.

TRAIN is warm-up context only. Evaluate exclusively on DEVELOPMENT.
This is a transaction-level benchmark, not a combined-engine comparison.
Run: python3 -m src.benchmark_scenarios_v3
"""
from pathlib import Path
import argparse
import pandas as pd

from src.tune_scenarios_v22 import load_period, build_history, evaluate_flag
from src.scenario_features_v3 import sender_rolling_features


def metrics_for(history, window_days, typology, mask_fn):
    tx = sender_rolling_features(history, window_days)
    return evaluate_flag(tx, mask_fn(tx), typology)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/temporal/SAML-D_train.csv"))
    parser.add_argument("--development", type=Path, default=Path("data/temporal/SAML-D_development.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/scenario_tuning_v3/rolling_benchmark.csv"))
    parser.add_argument("--warmup-days", type=int, default=120)
    args = parser.parse_args()

    train = load_period(args.train, "train")
    development = load_period(args.development, "development")
    history, dev_start = build_history(train, development, args.warmup_days)
    history = history.sort_values("Transaction_timestamp", kind="stable")
    if train["Transaction_timestamp"].max() >= dev_start:
        raise ValueError("TRAIN must precede DEVELOPMENT chronologically")
    print(f"DEVELOPMENT rows: {len(development):,}; history rows: {len(history):,}", flush=True)
    print("HOLDOUT is never loaded.", flush=True)

    configs = [
        ("Smurfing", 90, lambda tx: (
            tx["tx_count"].ge(3)
            & tx["median_amount"].lt(3000)
            & tx["aggregate_amount"].ge(10000)
            & tx["unique_receivers"].le(3)
            & tx["cross_border_rate"].le(.15)
            & tx["currency_mismatch_rate"].le(.15)
        )),
        ("Fan_Out", 21, lambda tx: (
            tx["unique_receivers"].ge(3)
            & tx["tx_count"].between(3, 12)
            & (tx["cross_border_rate"].ge(.30)
               | tx["currency_mismatch_rate"].ge(.30))
        )),
    ]
    rows = []
    for typology, days, mask in configs:
        print(f"Computing {typology} with {days}-day rolling window...", flush=True)
        result = metrics_for(history, days, typology, mask)
        rows.append({"Scenario": typology, "Window_days": days, **result})
        print(f"Alert rate: {result['Trigger_rate']:.4%}; "
              f"precision: {result['Precision']:.4%}; "
              f"typology recall: {result[f'{typology}_recall']:.4%}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved benchmark to {args.output}", flush=True)


if __name__ == "__main__":
    main()
