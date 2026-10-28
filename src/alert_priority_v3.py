"""Explainable prioritization of existing V3 scenario flags.

The score is a transparent queue-ordering heuristic, not a probability.
This module does not modify detection rules or evaluation metrics.
"""
import pandas as pd

SCENARIO_DESCRIPTIONS = {
    "Smurfing": "Repeated lower-value transfers and concentrated sender activity",
    "Fan_Out": "One sender transfers to multiple counterparties",
    "Structuring": "Receiver-side aggregation proxy; cash transaction status unverified",
    "Fan_In": "Multiple senders transfer to a common receiver",
}


def prioritize_alerts(transactions, flags):
    """Return flagged rows with reason codes and transparent priority scores.

    transactions and flags must share the same unique index. Flags must
    contain all four V3 scenario names, with boolean values.
    """
    required = list(SCENARIO_DESCRIPTIONS)
    if not transactions.index.is_unique or not flags.index.is_unique:
        raise ValueError("Unique transaction indices are required")
    if not transactions.index.equals(flags.index):
        raise ValueError("Transaction and flag indices must match")
    if not set(required).issubset(flags.columns):
        raise ValueError("Missing V3 scenario flags")
    selected = flags[required].fillna(False).astype(bool)
    selected = selected.loc[selected.any(axis=1)]
    output = transactions.loc[selected.index].copy()
    output["Scenario_count"] = selected.sum(axis=1).astype(int)
    output["Triggered_scenarios"] = selected.apply(
        lambda r: ";".join(name for name in required if r[name]), axis=1)
    output["Alert_reason"] = output["Triggered_scenarios"].map(
        lambda s: " | ".join(SCENARIO_DESCRIPTIONS[n] for n in s.split(";")))
    # Score uses only scenario concurrence and present transaction flags.
    border = output["Cross_border"].astype(int) if "Cross_border" in output else 0
    mismatch = output["Currency_mismatch"].astype(int) if "Currency_mismatch" in output else 0
    output["Priority_score"] = (
        40 + 15 * (output["Scenario_count"] - 1) + 10 * border + 10 * mismatch
    ).clip(upper=100)
    output["Priority"] = pd.cut(
        output["Priority_score"], [-1, 49, 69, 100],
        labels=["LOW", "MEDIUM", "HIGH"])
    return output.sort_values("Priority_score", ascending=False, kind="stable")
