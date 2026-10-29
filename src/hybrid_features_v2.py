"""Unsupervised, label-free scenario features for hybrid AML models.

Consumes precomputed *as-of-event* behavioral features. This function does not
calculate rolling histories: callers must prove those input features are causal.
No AML labels are read or used for scoring.
"""
from __future__ import annotations

import pandas as pd

from src.rule_based import SCENARIO_WEIGHTS, ScenarioConfig, apply_scenarios

SCENARIO_FEATURES = tuple(SCENARIO_WEIGHTS)
NUMERIC_FEATURES = (
    "Scenario_count", "Scenario_score", "Sender_24h_tx_count",
    "Sender_24h_amount", "Sender_prior_tx_count",
    "Sender_seconds_since_previous", "Sender_prior_avg_amount",
    "Sender_window_tx_count", "Sender_window_amount",
    "Sender_window_unique_receivers", "Receiver_window_tx_count",
    "Receiver_window_amount", "Receiver_window_unique_senders",
)
REQUIRED_BASE = (
    "Amount", "Max_country_risk", "Any_sanctioned_country",
    "Cross_border", "Currency_mismatch", "Sender_seconds_since_previous",
    "Sender_prior_tx_count", "Sender_prior_avg_amount",
)


def build_hybrid_features(data: pd.DataFrame,
                          config: ScenarioConfig = ScenarioConfig()) -> pd.DataFrame:
    """Return model inputs; never use labels or the final rule alert as a target."""
    missing = sorted(set(REQUIRED_BASE) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required as-of-event columns: {missing}")
    network = {
        "Sender_window_tx_count", "Sender_window_amount",
        "Sender_window_unique_receivers", "Receiver_window_tx_count",
        "Receiver_window_amount", "Receiver_window_unique_senders",
    }
    if 0 < len(network.intersection(data.columns)) < len(network):
        raise ValueError("Partial network feature set; provide all six or none")
    scored = apply_scenarios(data, config)
    names = list(SCENARIO_FEATURES) + [
        name for name in NUMERIC_FEATURES if name in scored.columns
    ]
    return scored.loc[:, names].copy()
