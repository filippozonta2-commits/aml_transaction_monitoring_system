"""Small DEVELOPMENT-only threshold sweep using causal V3 rolling features.

Computes rolling feature tables only once per window size. HOLDOUT is not loaded.
"""
from pathlib import Path
import argparse
import pandas as pd

from src.scenario_features_v3 import sender_rolling_features
from src.tune_scenarios_v22 import load_period, build_history, evaluate_flag, shortlist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/temporal/SAML-D_train.csv"))
    parser.add_argument("--development", type=Path, default=Path("data/temporal/SAML-D_development.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/scenario_tuning_v3"))
    args = parser.parse_args()

    train = load_period(args.train, "train")
    dev = load_period(args.development, "development")
    history, start = build_history(train, dev, 120)
    history = history.sort_values("Transaction_timestamp", kind="stable")
    if train["Transaction_timestamp"].max() >= start:
        raise ValueError("TRAIN must precede DEVELOPMENT")
    rows = {"Smurfing": [], "Fan_Out": []}
    for typology, days in [("Smurfing", 60), ("Smurfing", 90), ("Fan_Out", 10), ("Fan_Out", 21)]:
        print(f"Rolling features: {typology}, {days} days", flush=True)
        tx = sender_rolling_features(history, days)
        if typology == "Smurfing":
            for min_count in [3, 5]:
                for median_ceiling in [2000, 3000, 4000]:
                    for aggregate_floor in [10000, 20000]:
                        mask = (tx["tx_count"].ge(min_count)
                                & tx["median_amount"].lt(median_ceiling)
                                & tx["aggregate_amount"].ge(aggregate_floor)
                                & tx["unique_receivers"].le(3)
                                & tx["cross_border_rate"].le(.15)
                                & tx["currency_mismatch_rate"].le(.15))
                        rows[typology].append({
                            "Window_days": days, "Min_count": min_count,
                            "Median_ceiling": median_ceiling,
                            "Aggregate_floor": aggregate_floor,
                            **evaluate_flag(tx, mask, typology),
                        })
        else:
            for min_cp in [3, 4, 5]:
                for max_tx in [8, 12, 16]:
                    for risk in [.2, .3, .4]:
                        mask = (tx["unique_receivers"].ge(min_cp)
                                & tx["tx_count"].between(min_cp, max_tx)
                                & (tx["cross_border_rate"].ge(risk)
                                   | tx["currency_mismatch_rate"].ge(risk)))
                        rows[typology].append({
                            "Window_days": days, "Min_counterparties": min_cp,
                            "Max_transactions": max_tx, "Geo_FX_threshold": risk,
                            **evaluate_flag(tx, mask, typology),
                        })
        del tx

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for typology, values in rows.items():
        frame = pd.DataFrame(values)
        path = args.output_dir / f"{typology.lower()}_grid.csv"
        frame.to_csv(path, index=False)
        cap = .02 if typology == "Fan_Out" else .01
        eligible = frame.loc[frame["Trigger_rate"].le(cap)]
        print(f"\\n=== {typology}: best configs at <= {cap:.0%} alert rate ===", flush=True)
        if eligible.empty:
            print("No configuration meets alert-rate cap.", flush=True)
        else:
            print(eligible.sort_values([f"{typology}_recall", "Precision", "Trigger_rate"],
                                       ascending=[False, False, True]).head(10).to_string(index=False), flush=True)
        print(f"Saved {len(frame)} configurations to {path}", flush=True)


if __name__ == "__main__":
    main()
