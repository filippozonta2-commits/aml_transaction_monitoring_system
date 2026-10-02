"""Unified AML scenario catalogue built from frozen DEVELOPMENT selections.

This module is deliberately configuration-first. It exposes one stable catalogue
for downstream execution/reporting while keeping scenario thresholds out of the
dashboard layer. HOLDOUT evaluation belongs in a separate one-shot runner.
"""

from scenario_config_frozen import FROZEN_SCENARIOS, assert_holdout_safe

DISPLAY_NAMES = {
    "smurfing_cash_deposit": "Smurfing - Cash Deposit",
    "cash_withdrawal": "Cash Withdrawal",
    "fan_out": "Fan-Out",
    "deposit_send": "Deposit-Send",
    "structuring": "Structuring",
    "fan_in": "Fan-In",
}

TARGET_TYPOLOGIES = {
    "smurfing_cash_deposit": "Smurfing",
    "cash_withdrawal": "Cash_Withdrawal",
    "fan_out": "Fan_Out",
    "deposit_send": "Deposit-Send",
    "structuring": "Structuring",
    "fan_in": "Fan_In",
}


def frozen_catalogue():
    """Return the frozen scenario catalogue and fail if anything is unfrozen."""
    assert_holdout_safe()
    pending = {
        name: cfg.get("status")
        for name, cfg in FROZEN_SCENARIOS.items()
        if cfg.get("status") != "frozen_development"
    }
    if pending:
        raise RuntimeError(f"Scenario catalogue is not fully frozen: {pending}")

    return {
        name: {
            "display_name": DISPLAY_NAMES[name],
            "target_typology": TARGET_TYPOLOGIES[name],
            **cfg,
        }
        for name, cfg in FROZEN_SCENARIOS.items()
    }


def print_catalogue():
    cat = frozen_catalogue()
    print("\n=== FROZEN AML SCENARIO CATALOGUE ===")
    for name, cfg in cat.items():
        m = cfg["development_metrics"]
        recall = next(
            (v for k, v in m.items() if k.endswith("_recall") and "account" not in k),
            m.get("target_recall"),
        )
        print(
            f"{cfg['display_name']:<26} "
            f"status={cfg['status']:<18} "
            f"precision={m.get('precision', float('nan')):.4%} "
            f"target_recall={recall:.2%}"
        )
    print(f"\nFrozen scenarios: {len(cat)}")
    print("HOLDOUT has not been accessed by this catalogue.")


if __name__ == "__main__":
    print_catalogue()
