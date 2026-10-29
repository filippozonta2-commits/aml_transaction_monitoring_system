"""Train and evaluate Hybrid AML V2 on prepared, causally engineered split CSVs.

This does NOT reconstruct historical V1 and does NOT create event-time features.
Requires explicit independent train/development/test files and an untouched test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from xgboost import XGBClassifier

from src.hybrid_features_v2 import build_hybrid_features

LABEL = "Is_laundering"
TIME = "event_time"
ID = "transaction_id"
RAW = ("Amount", "Max_country_risk", "Cross_border", "Currency_mismatch",
       "Any_sanctioned_country")
MODES = ("raw", "scenario", "hybrid")


def read_split(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    for column in (ID, TIME, LABEL, *RAW):
        if column not in data:
            raise ValueError(f"{path}: missing {column}")
    if data[[ID, TIME, LABEL]].isna().any().any():
        raise ValueError(f"{path}: missing IDs, times or labels")
    if data[ID].duplicated().any():
        raise ValueError(f"{path}: duplicate IDs")
    if not data[LABEL].isin([0, 1]).all():
        raise ValueError(f"{path}: nonbinary labels")
    data[TIME] = pd.to_datetime(data[TIME], utc=True, errors="raise")
    if data[TIME].isna().any() or not data[TIME].is_monotonic_increasing:
        raise ValueError(f"{path}: invalid or unsorted timestamps")
    return data


def check_splits(train, dev, test):
    if not (train[TIME].max() < dev[TIME].min()
            and dev[TIME].max() < test[TIME].min()):
        raise ValueError("Chronological split overlap")
    ids = [set(frame[ID].astype(str)) for frame in (train, dev, test)]
    if any(ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Transaction ID overlap")
    for frame in (train, dev, test):
        if frame[LABEL].nunique() != 2:
            raise ValueError("Each split must contain both classes")


def feature_matrix(data, mode):
    raw = data.loc[:, RAW].copy()
    scenario = build_hybrid_features(data)
    if mode == "raw":
        out = raw
    elif mode == "scenario":
        out = scenario
    elif mode == "hybrid":
        out = pd.concat([raw, scenario], axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
    else:
        raise ValueError(mode)
    out = out.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    if out.isna().all().any():
        raise ValueError("Entirely missing model feature")
    return out


def choose_threshold(y, probabilities, max_alert_rate):
    if not 0 < max_alert_rate < 1:
        raise ValueError("max_alert_rate must be in (0,1)")
    # Rank-based budget selection; ties may cause fewer alerts than the budget.
    budget = max(1, int(np.floor(len(probabilities) * max_alert_rate)))
    ordered = np.sort(np.asarray(probabilities, dtype=float))[::-1]
    threshold = float(ordered[budget - 1])
    # Strictly above the boundary guarantees the alert budget under ties.
    cutoff = float(np.nextafter(threshold, np.inf))
    selected = np.asarray(probabilities) >= cutoff
    if selected.sum() > budget:
        raise AssertionError("Alert budget exceeded")
    return cutoff


def report(y, probabilities, threshold):
    flags = np.asarray(probabilities) >= threshold
    return {
        "rows": int(len(y)),
        "positives": int(np.sum(y)),
        "flagged": int(flags.sum()),
        "true_positives": int(np.sum(flags & (np.asarray(y) == 1))),
        "precision": float(precision_score(y, flags, zero_division=0)),
        "recall": float(recall_score(y, flags, zero_division=0)),
        "alert_rate": float(flags.mean()),
        "average_precision": float(average_precision_score(y, probabilities)),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    paths = (args.train, args.development, args.test)
    train, dev, test = [read_split(path) for path in paths]
    check_splits(train, dev, test)
    results = {
        "experimental": True,
        "test_used_for_model_selection": False,
        "data_sha256": {name: sha256(path) for name, path in
                        zip(("train", "development", "test"), paths)},
        "max_development_alert_rate": args.max_alert_rate,
        "models": {},
    }
    for mode in MODES:
        x_train, x_dev, x_test = [feature_matrix(frame, mode)
                                 for frame in (train, dev, test)]
        if list(x_train.columns) != list(x_dev.columns) or list(x_train.columns) != list(x_test.columns):
            raise ValueError(f"{mode}: inconsistent feature schemas")
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=4, tree_method="hist", eval_metric="logloss",
        )
        model.fit(x_train, train[LABEL])
        dev_probs = model.predict_proba(x_dev)[:, 1]
        threshold = choose_threshold(dev[LABEL], dev_probs, args.max_alert_rate)
        test_probs = model.predict_proba(x_test)[:, 1]
        results["models"][mode] = {
            "features": list(x_train.columns),
            "threshold_selected_on_development": threshold,
            "development": report(dev[LABEL], dev_probs, threshold),
            "test": report(test[LABEL], test_probs, threshold),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/hybrid_v2/experiment.json"))
    parser.add_argument("--max-alert-rate", type=float, default=0.02)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
