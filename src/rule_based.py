"""Transparent rule-based AML alerting engine.

The engine is intentionally separate from the machine-learning pipeline.
Each rule creates an interpretable flag and contributes to an aggregate
risk score. Quantitative thresholds are configurable and should be calibrated
using training/reference data rather than test labels.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleConfig:
    high_amount: float = 10_000.0
    high_country_risk: float = 7.0
    rapid_repeat_seconds: float = 3_600.0
    high_prior_tx_count: int = 10
    alert_score_threshold: int = 40


RULE_WEIGHTS = {
    "Rule_high_amount": 20,
    "Rule_cross_border": 10,
    "Rule_currency_mismatch": 10,
    "Rule_high_risk_country": 20,
    "Rule_sanctioned_country": 30,
    "Rule_rapid_sender_activity": 20,
    "Rule_high_sender_activity": 15,
}


def apply_rules(data: pd.DataFrame, config: RuleConfig = RuleConfig()) -> pd.DataFrame:
    """Apply interpretable AML rules to an engineered transaction dataset."""
    result = data.copy()

    result["Rule_high_amount"] = result["Amount"].ge(config.high_amount).astype(int)
    result["Rule_cross_border"] = result["Cross_border"].fillna(0).astype(int)
    result["Rule_currency_mismatch"] = (
        result["Currency_mismatch"].fillna(0).astype(int)
    )
    result["Rule_high_risk_country"] = (
        result["Max_country_risk"].ge(config.high_country_risk).fillna(False).astype(int)
    )
    result["Rule_sanctioned_country"] = (
        result["Any_sanctioned_country"].fillna(0).astype(int)
    )
    result["Rule_rapid_sender_activity"] = (
        result["Sender_seconds_since_previous"]
        .between(0, config.rapid_repeat_seconds)
        .fillna(False)
        .astype(int)
    )
    result["Rule_high_sender_activity"] = (
        result["Sender_prior_tx_count"]
        .ge(config.high_prior_tx_count)
        .fillna(False)
        .astype(int)
    )

    score = np.zeros(len(result), dtype=int)
    for rule, weight in RULE_WEIGHTS.items():
        score += result[rule].to_numpy(dtype=int) * weight

    result["Rule_score"] = np.clip(score, 0, 100)
    result["Rule_alert"] = (
        result["Rule_score"] >= config.alert_score_threshold
    ).astype(int)

    result["Alert_reasons"] = result.apply(_alert_reasons, axis=1)
    return result


def _alert_reasons(row: pd.Series) -> str:
    """Return human-readable reasons for triggered rules."""
    labels = {
        "Rule_high_amount": "High transaction amount",
        "Rule_cross_border": "Cross-border transaction",
        "Rule_currency_mismatch": "Currency mismatch",
        "Rule_high_risk_country": "High-risk jurisdiction",
        "Rule_sanctioned_country": "Sanctioned-jurisdiction exposure",
        "Rule_rapid_sender_activity": "Rapid sender activity",
        "Rule_high_sender_activity": "High sender transaction activity",
    }
    reasons = [label for rule, label in labels.items() if row.get(rule, 0) == 1]
    return "; ".join(reasons)


def rule_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize rule frequency and AML hit rate for analysis."""
    rows = []
    for rule in RULE_WEIGHTS:
        triggered = data[data[rule] == 1]
        rows.append(
            {
                "Rule": rule,
                "Triggered": len(triggered),
                "Trigger_rate": len(triggered) / len(data) if len(data) else 0,
                "AML_cases": int(triggered["Is_laundering"].sum())
                if "Is_laundering" in triggered
                else np.nan,
                "AML_rate_when_triggered": triggered["Is_laundering"].mean()
                if "Is_laundering" in triggered and len(triggered)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("Triggered", ascending=False)
