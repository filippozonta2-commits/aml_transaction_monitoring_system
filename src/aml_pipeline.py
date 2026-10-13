"""Reproducible AML baseline pipeline for SAML-D style CSV data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_column(frame, candidates):
    normalized = {str(c).lower().replace(" ", "_"): c for c in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def prepare(frame):
    label = find_column(frame, ["is_laundering", "is_aml", "label", "target"])
    date = find_column(frame, ["date", "timestamp", "transaction_date", "datetime"])
    if label is None or date is None:
        raise ValueError("Expected a laundering label and transaction timestamp column")
    data = frame.copy()
    data["_time"] = pd.to_datetime(data[date], errors="coerce")
    data = data.dropna(subset=["_time", label]).sort_values("_time").reset_index(drop=True)
    data["_target"] = data[label].astype(int)
    if not set(data["_target"].unique()).issubset({0, 1}):
        raise ValueError("Label must be binary 0/1")
    amount = find_column(data, ["amount", "amount_paid", "transaction_amount"])
    if amount:
        data["log_amount"] = np.log1p(pd.to_numeric(data[amount], errors="coerce").clip(lower=0))
    sender = find_column(data, ["sender_bank_location", "sender_country"])
    receiver = find_column(data, ["receiver_bank_location", "receiver_country"])
    if sender and receiver:
        data["cross_border"] = (data[sender].astype(str) != data[receiver].astype(str)).astype(int)
    features = [c for c in ["log_amount", "cross_border"] if c in data]
    for options in [
        ["payment_type", "payment_method"],
        ["payment_currency", "currency"],
        ["received_currency", "receiving_currency"],
        ["sender_bank_location", "sender_country"],
        ["receiver_bank_location", "receiver_country"],
    ]:
        col = find_column(data, options)
        if col and col not in features:
            features.append(col)
    if not features:
        raise ValueError("No supported transaction features found")
    return data, features


def split_temporally(data):
    n = len(data)
    if n < 30:
        raise ValueError("At least 30 records required")
    train_end, dev_end = int(n * .6), int(n * .8)
    # Keep identical timestamps in the same partition to prevent time overlap.
    while train_end < n and data.loc[train_end, "_time"] == data.loc[train_end - 1, "_time"]:
        train_end += 1
    dev_end = max(dev_end, train_end + 1)
    while dev_end < n and data.loc[dev_end, "_time"] == data.loc[dev_end - 1, "_time"]:
        dev_end += 1
    if dev_end >= n or train_end >= dev_end:
        raise ValueError("Insufficient distinct timestamps for temporal splits")
    return data.iloc[:train_end], data.iloc[train_end:dev_end], data.iloc[dev_end:]


def score(y, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    result = {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "alert_rate": float(predicted.mean()),
        "alerts": int(predicted.sum()),
    }
    result["average_precision"] = float(average_precision_score(y, probabilities)) if y.nunique() > 1 else None
    result["roc_auc"] = float(roc_auc_score(y, probabilities)) if y.nunique() > 1 else None
    return result


def run(input_path, output_dir):
    data, features = prepare(pd.read_csv(input_path))
    train, dev, test = split_temporally(data)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
    categorical = [c for c in features if c not in numeric]
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("features", transformer), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    if train["_target"].nunique() < 2:
        raise ValueError("Training partition must contain both classes")
    model.fit(train[features], train["_target"])
    dev_scores = model.predict_proba(dev[features])[:, 1]
    # Threshold selection is performed on dev only, never on holdout.
    thresholds = np.unique(np.quantile(dev_scores, np.linspace(0, 1, 101)))
    threshold = max(thresholds, key=lambda t: (2 * score(dev["_target"], dev_scores, t)["precision"] * score(dev["_target"], dev_scores, t)["recall"] / max(score(dev["_target"], dev_scores, t)["precision"] + score(dev["_target"], dev_scores, t)["recall"], 1e-12)))
    test_scores = model.predict_proba(test[features])[:, 1]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {"features": features, "split_sizes": {"train": len(train), "dev": len(dev), "test": len(test)}, "threshold": float(threshold), "dev": score(dev["_target"], dev_scores, threshold), "test": score(test["_target"], test_scores, threshold)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"timestamp": test["_time"], "actual": test["_target"], "risk_score": test_scores, "alert": (test_scores >= threshold).astype(int)}).to_csv(output / "holdout_alerts.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/pipeline")
    args = parser.parse_args()
    run(args.input, args.output)
