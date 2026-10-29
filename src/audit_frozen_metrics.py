"""Reconcile published frozen HOLDOUT counts; optionally audit row-level evidence.

This is an audit utility, NOT the missing training/evaluation pipeline.
Summary mode validates arithmetic only; it cannot verify the original experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HOLDOUT_N = 1_898_009
AML_N = 2_136
ML_THRESHOLD = 0.653366
EXPECTED = {
    "Rule-Based V3": {"alerts": 127_981, "hits": 1_225},
    "XGBoost V1": {"alerts": 39_409, "hits": 1_798},
}


def metrics(n: int, positives: int, alerts: int, hits: int) -> dict:
    """Derive a complete confusion matrix from integer transaction counts."""
    values = (n, positives, alerts, hits)
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("Counts must be nonnegative integers")
    if not (hits <= positives <= n and hits <= alerts <= n):
        raise ValueError("Inconsistent positive/alert/hit counts")
    fp, fn = alerts - hits, positives - hits
    tn = n - hits - fp - fn
    if tn < 0:
        raise ValueError("Negative true-negative count")
    return {
        "n": n, "positives": positives, "alerts": alerts, "tp": hits,
        "fp": fp, "fn": fn, "tn": tn,
        "precision": hits / alerts if alerts else 0.0,
        "recall": hits / positives if positives else 0.0,
        "alert_rate": alerts / n if n else 0.0,
        "prevalence": positives / n if n else 0.0,
    }


def audit_summary(path: Path) -> dict:
    """Check arithmetic of the published mapping CSV, not raw HOLDOUT rows."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        records = {r["Metric"]: r for r in csv.DictReader(f)}
    output = {}
    for name in EXPECTED:
        n = int(records["HOLDOUT transactions"][name])
        positives = int(records["AML positives"][name])
        alerts = int(records["Flagged transactions"][name])
        hits = int(records["AML hits"][name])
        m = metrics(n, positives, alerts, hits)
        if (n, positives, alerts, hits) != (
            HOLDOUT_N, AML_N, EXPECTED[name]["alerts"], EXPECTED[name]["hits"]
        ):
            raise ValueError(f"{name}: mapping CSV differs from frozen published counts")
        for label, key in (("Precision", "precision"), ("Recall", "recall"),
                           ("Alert rate", "alert_rate")):
            reported = float(records[label][name])
            if abs(reported - m[key]) > 0.000002:
                raise ValueError(f"{name}: inconsistent {label}: {reported} vs {m[key]}")
        output[name] = m
    return {"evidence_level": "published-summary-arithmetic-only", "detectors": output,
            "pr_auc": None, "warning": "No row-level labels/scores were independently audited."}


def audit_rows(path: Path) -> dict:
    """Audit a joined, one-row-per-HOLDOUT-transaction evidence CSV.

    Required: Is_laundering, RULE_ALERT_V3, ML_ALERT_V1, ml_probability.
    Optional: holdout_row_id (checked for uniqueness). This function requires
    *actual* row-level data and never fabricates scores or a PR-AUC.
    """
    import pandas as pd
    from sklearn.metrics import average_precision_score

    df = pd.read_csv(path)
    required = ["Is_laundering", "RULE_ALERT_V3", "ML_ALERT_V1", "ml_probability"]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Null labels, flags or scores")
    for c in required[:3]:
        if not df[c].isin([0, 1]).all():
            raise ValueError(f"Nonbinary column: {c}")
    if not df["ml_probability"].between(0, 1).all():
        raise ValueError("ML probabilities outside [0,1]")
    if "holdout_row_id" in df and (df["holdout_row_id"].isna().any() or
                                   df["holdout_row_id"].duplicated().any()):
        raise ValueError("Missing/duplicate HOLDOUT row IDs")
    expected_flags = df["ml_probability"].ge(ML_THRESHOLD).astype(int)
    if not expected_flags.eq(df["ML_ALERT_V1"]).all():
        raise ValueError("ML flags do not match the documented frozen threshold")
    n, positives = len(df), int(df["Is_laundering"].sum())
    if (n, positives) != (HOLDOUT_N, AML_N):
        raise ValueError(f"Unexpected HOLDOUT population: {(n, positives)}")
    output = {}
    for name, flag in (("Rule-Based V3", "RULE_ALERT_V3"),
                       ("XGBoost V1", "ML_ALERT_V1")):
        alerts = int(df[flag].sum())
        hits = int((df[flag] * df["Is_laundering"]).sum())
        if (alerts, hits) != (EXPECTED[name]["alerts"], EXPECTED[name]["hits"]):
            raise ValueError(f"{name}: row-level counts differ from frozen benchmark")
        output[name] = metrics(n, positives, alerts, hits)
    output["XGBoost V1"]["pr_auc_average_precision"] = float(
        average_precision_score(df["Is_laundering"], df["ml_probability"])
    )
    return {"evidence_level": "joined-row-level-reconciled", "detectors": output,
            "threshold": ML_THRESHOLD}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path,
                        default=Path("docs/mapping/04_holdout_evaluation.csv"))
    parser.add_argument("--joined-row-level", type=Path,
                        help="Optional joined HOLDOUT labels, flags and ML scores")
    args = parser.parse_args()
    report = audit_rows(args.joined_row_level) if args.joined_row_level else audit_summary(args.mapping)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
