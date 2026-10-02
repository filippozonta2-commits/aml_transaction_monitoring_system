"""Configurable scenario-based AML transaction-monitoring engine.

This module implements original portfolio scenarios inspired by common
transaction-monitoring patterns. It does not reproduce proprietary vendor or
institution configurations. Scenario thresholds are explicit and should be
calibrated on training/reference data before final evaluation.
"""

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioConfig:
    # Transaction-level scenarios
    single_large_amount: float = 50_000.0
    high_country_risk: float = 7.0

    # Behavioral scenarios
    rapid_repeat_seconds: float = 3_600.0
    high_velocity_prior_count: int = 10
    unusual_amount_multiplier: float = 3.0

    # Aggregate / network-oriented scenarios
    structuring_amount_ceiling: float = 10_000.0
    structuring_prior_count: int = 5
    fan_activity_prior_count: int = 8

    # Alert aggregation
    scenario_alert_threshold: int = 1


SCENARIO_WEIGHTS: Dict[str, int] = {
    "SCN_SINGLE_LARGE": 60,
    "SCN_HIGH_RISK_GEOGRAPHY": 40,
    "SCN_SANCTIONED_GEOGRAPHY": 80,
    "SCN_CROSS_BORDER_CURRENCY_MISMATCH": 30,
    "SCN_RAPID_REPEAT_ACTIVITY": 40,
    "SCN_HIGH_VELOCITY": 45,
    "SCN_UNUSUAL_AMOUNT": 50,
    "SCN_STRUCTURING_PROXY": 50,
    "SCN_FAN_ACTIVITY_PROXY": 35,
}


SCENARIO_LABELS = {
    "SCN_SINGLE_LARGE": "Single large transaction",
    "SCN_HIGH_RISK_GEOGRAPHY": "High-risk geography",
    "SCN_SANCTIONED_GEOGRAPHY": "Sanctioned-jurisdiction exposure",
    "SCN_CROSS_BORDER_CURRENCY_MISMATCH": "Cross-border currency mismatch",
    "SCN_RAPID_REPEAT_ACTIVITY": "Rapid repeat sender activity",
    "SCN_HIGH_VELOCITY": "High sender transaction velocity",
    "SCN_UNUSUAL_AMOUNT": "Amount materially above sender history",
    "SCN_STRUCTURING_PROXY": "Potential structuring pattern",
    "SCN_FAN_ACTIVITY_PROXY": "Potential fan activity",
}


def apply_scenarios(
    data: pd.DataFrame, config: ScenarioConfig = ScenarioConfig()
) -> pd.DataFrame:
    """Apply configurable AML scenarios to engineered transaction data."""
    result = data.copy()

    result["SCN_SINGLE_LARGE"] = (
        result["Amount"].ge(config.single_large_amount).astype(int)
    )

    result["SCN_HIGH_RISK_GEOGRAPHY"] = (
        result["Max_country_risk"]
        .ge(config.high_country_risk)
        .fillna(False)
        .astype(int)
    )

    result["SCN_SANCTIONED_GEOGRAPHY"] = (
        result["Any_sanctioned_country"].fillna(0).astype(int)
    )

    result["SCN_CROSS_BORDER_CURRENCY_MISMATCH"] = (
        result["Cross_border"].fillna(0).astype(bool)
        & result["Currency_mismatch"].fillna(0).astype(bool)
    ).astype(int)

    result["SCN_RAPID_REPEAT_ACTIVITY"] = (
        result["Sender_seconds_since_previous"]
        .between(0, config.rapid_repeat_seconds)
        .fillna(False)
        .astype(int)
    )

    result["SCN_HIGH_VELOCITY"] = (
        result["Sender_prior_tx_count"]
        .ge(config.high_velocity_prior_count)
        .fillna(False)
        .astype(int)
    )

    historical_average = result["Sender_prior_avg_amount"].replace(0, np.nan)
    result["SCN_UNUSUAL_AMOUNT"] = (
        result["Amount"]
        .ge(historical_average * config.unusual_amount_multiplier)
        .fillna(False)
        .astype(int)
    )

    # Proxy until rolling-window aggregation is added. This deliberately avoids
    # claiming exact structuring detection from transaction-level information.
    result["SCN_STRUCTURING_PROXY"] = (
        result["Amount"].lt(config.structuring_amount_ceiling)
        & result["Sender_prior_tx_count"].ge(config.structuring_prior_count)
        & result["Sender_seconds_since_previous"]
        .between(0, config.rapid_repeat_seconds)
        .fillna(False)
    ).astype(int)

    # Proxy for repeated sender activity. True fan-out detection will use
    # rolling unique-counterparty counts in the next network feature layer.
    result["SCN_FAN_ACTIVITY_PROXY"] = (
        result["Sender_prior_tx_count"]
        .ge(config.fan_activity_prior_count)
        .fillna(False)
        .astype(int)
    )

    scenario_columns = list(SCENARIO_WEIGHTS)
    result["Scenario_count"] = result[scenario_columns].sum(axis=1)

    weighted_score = np.zeros(len(result), dtype=int)
    for scenario, weight in SCENARIO_WEIGHTS.items():
        weighted_score += result[scenario].to_numpy(dtype=int) * weight

    result["Scenario_score"] = np.clip(weighted_score, 0, 100)
    result["Scenario_alert"] = (
        result["Scenario_count"] >= config.scenario_alert_threshold
    ).astype(int)
    result["Alert_reasons"] = result.apply(_scenario_reasons, axis=1)

    return result


def _scenario_reasons(row: pd.Series) -> str:
    triggered = [
        label
        for scenario, label in SCENARIO_LABELS.items()
        if row.get(scenario, 0) == 1
    ]
    return "; ".join(triggered)


def scenario_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize scenario volume and observed AML rate."""
    rows = []

    for scenario in SCENARIO_WEIGHTS:
        triggered = data[data[scenario] == 1]
        rows.append(
            {
                "Scenario": scenario,
                "Description": SCENARIO_LABELS[scenario],
                "Triggered": len(triggered),
                "Trigger_rate": len(triggered) / len(data) if len(data) else 0.0,
                "AML_cases": int(triggered["Is_laundering"].sum())
                if "Is_laundering" in data.columns
                else np.nan,
                "AML_rate_when_triggered": triggered["Is_laundering"].mean()
                if "Is_laundering" in data.columns and len(triggered)
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("Triggered", ascending=False)


def scenario_configuration(config: ScenarioConfig = ScenarioConfig()) -> pd.DataFrame:
    """Expose scenario parameters for reporting/dashboard configuration."""
    return pd.DataFrame(
        [{"Parameter": key, "Value": value} for key, value in asdict(config).items()]
    )
