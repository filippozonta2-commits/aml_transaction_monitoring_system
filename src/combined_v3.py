"""Combined causal V3 scenario evaluation on DEVELOPMENT."""
import pandas as pd
from src.tune_scenarios_v22 import load_period, build_history
from src.scenario_features_v3 import sender_rolling_features


def main():
    train = load_period("data/temporal/SAML-D_train.csv", "train")
    dev = load_period("data/temporal/SAML-D_development.csv", "development")
    history, start = build_history(train, dev, 120)
    if train["Transaction_timestamp"].max() >= start:
        raise ValueError("TRAIN must precede DEVELOPMENT")
    history = history.sort_values("Transaction_timestamp", kind="stable")
    idx = history.index[history["Period"].eq("development")]
    labels = history.loc[idx, "Is_laundering"].eq(1)
    print("Computing Smurfing 90-day features...", flush=True)
    sm = sender_rolling_features(history, 90)
    sm_flag = (
        sm["tx_count"].ge(3) & sm["median_amount"].lt(3000)
        & sm["aggregate_amount"].ge(20000)
        & sm["unique_receivers"].le(3)
        & sm["cross_border_rate"].le(.15)
        & sm["currency_mismatch_rate"].le(.15)
    ).loc[idx]
    del sm
    print("Computing Fan-Out 21-day features...", flush=True)
    fo = sender_rolling_features(history, 21)
    rows = []
    for name, maximum, threshold in [
        ("recall", 12, .20),
        ("intermediate", 12, .30),
        ("selective", 8, .30),
    ]:
        fan_flag = (
            fo["unique_receivers"].ge(3)
            & fo["tx_count"].between(3, maximum)
            & (fo["cross_border_rate"].ge(threshold)
               | fo["currency_mismatch_rate"].ge(threshold))
        ).loc[idx]
        combined = sm_flag | fan_flag
        alerts = int(combined.sum())
        hits = int((combined & labels).sum())
        rows.append({
            "FanOut_variant": name,
            "Smurfing_alerts": int(sm_flag.sum()),
            "FanOut_alerts": int(fan_flag.sum()),
            "Overlapping_alerts": int((sm_flag & fan_flag).sum()),
            "Combined_alerts": alerts,
            "AML_transactions": int(labels.sum()),
            "AML_hits": hits,
            "Alert_rate": alerts / len(idx),
            "Precision": hits / alerts if alerts else 0.,
            "Overall_AML_recall": hits / int(labels.sum()) if labels.sum() else 0.,
        })
    result = pd.DataFrame(rows)
    output = "results/scenario_tuning_v3/combined_v3.csv"
    from pathlib import Path
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print("Saved:", output, "| HOLDOUT not accessed.")


if __name__ == "__main__":
    main()
