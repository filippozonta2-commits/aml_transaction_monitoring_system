"""Frozen DEVELOPMENT-selected AML scenario configuration.

Do not tune these values on HOLDOUT. HOLDOUT is reserved for one-shot final
evaluation after the scenario catalogue is frozen.

Selection notes
---------------
Deposit-Send:
    D_cb_baseline from evaluate_deposit_send_v4.py.
Cash Withdrawal:
    Selected from the DEVELOPMENT shortlist as the higher-precision operating
    point while retaining ~67% transaction recall.
Smurfing:
    Cash-Deposit-aware operating point selected from DEVELOPMENT results.
Fan-Out:
    V2.2 selective operating point from DEVELOPMENT shortlist.

Structuring and Fan-In:
    Existing V2 definitions validated and frozen on DEVELOPMENT using
    freeze_structuring_fanin.py. No HOLDOUT data was accessed.
"""

FROZEN_SCENARIOS = {
    "smurfing_cash_deposit": {
        "status": "frozen_development",
        "payment_type": "Cash Deposit",
        "window_days": 45,
        "min_count": 3,
        "median_ceiling": 4000.0,
        "aggregate_floor": 10000.0,
        "development_metrics": {
            "triggered_transactions": 2204,
            "precision": 0.090290,
            "smurfing_recall": 0.925581,
            "smurfing_account_recall": 0.903226,
        },
    },
    "cash_withdrawal": {
        "status": "frozen_development",
        "payment_type": "Cash Withdrawal",
        "window_days": 7,
        "min_count": 5,
        "aggregate_floor": 300.0,
        "median_ceiling": 300.0,
        "development_metrics": {
            "triggered_transactions": 2188,
            "channel_trigger_rate": 0.036359,
            "precision": 0.086380,
            "cash_withdrawal_recall": 0.670213,
            "cash_withdrawal_account_recall": 0.717949,
        },
    },
    "fan_out": {
        "status": "frozen_development",
        "window_days": 21,
        "min_counterparties": 3,
        "max_transactions": 12,
        "geo_fx_threshold": 0.20,
        "development_metrics": {
            "triggered_transactions": 17261,
            "trigger_rate": 0.009087,
            "precision": 0.012108,
            "fan_out_recall": 0.739130,
            "fan_out_account_recall": 0.777778,
        },
    },
    "deposit_send": {
        "status": "frozen_development",
        "candidate": "D_cb_baseline",
        "horizon_hours": 72,
        "min_outflow_ratio": 0.50,
        "max_outflow_ratio": 3.0,
        "require_cross_border": True,
        "development_metrics": {
            "triggered_transactions": 24168,
            "trigger_rate": 0.012723167399041972,
            "precision": 0.004427341939755048,
            "lift_vs_overall_aml_rate": 4.234569764751806,
            "deposit_send_recall": 0.26285714285714284,
        },
    },
    "structuring": {
        "status": "frozen_development",
        "window_days": 10,
        "transaction_amount_ceiling": 10000.0,
        "min_unique_senders": 5,
        "aggregate_floor": 20000.0,
        "cross_border_rate_threshold": 0.20,
        "currency_mismatch_rate_threshold": 0.25,
        "geo_fx_logic": "OR",
        "development_metrics": {
            "triggered_transactions": 28520,
            "trigger_rate": 0.015014,
            "precision": 0.007679,
            "target_transactions": 335,
            "target_hits": 171,
            "target_recall": 0.510448,
        },
    },
    "fan_in": {
        "status": "frozen_development",
        "window_days": 10,
        "min_unique_senders": 5,
        "max_unique_senders": 15,
        "min_transactions": 5,
        "max_transactions": 20,
        "cross_border_rate_threshold": 0.10,
        "currency_mismatch_rate_threshold": 0.20,
        "geo_fx_logic": "OR",
        "development_metrics": {
            "triggered_transactions": 7607,
            "trigger_rate": 0.004005,
            "precision": 0.041541,
            "target_transactions": 113,
            "target_hits": 71,
            "target_recall": 0.628319,
        },
    },
}


def assert_holdout_safe():
    """Guardrail used by development runners before final holdout evaluation."""
    for name, cfg in FROZEN_SCENARIOS.items():
        if cfg["status"] == "frozen_development":
            assert "development_metrics" in cfg, name
