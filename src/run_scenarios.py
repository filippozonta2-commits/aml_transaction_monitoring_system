"""Run the AML scenario engine on a development dataset."""

from pathlib import Path
import argparse

import pandas as pd

from baseline_models import enrich_country_risk
from features import add_transaction_features, add_historical_account_features
from network_features import add_rolling_network_features
from rule_based import ScenarioConfig, apply_scenarios, scenario_summary


def run_scenarios(
    transactions_path: Path,
    country_risk_path: Path,
    output_dir: Path,
    lookback: str = "24h",
) -> None:
    print("Loading development transactions...")
    transactions = pd.read_csv(transactions_path)
    country_risk = pd.read_csv(country_risk_path)

    print(f"Transactions loaded: {len(transactions):,}")
    print(f"AML transactions: {transactions['Is_laundering'].sum():,}")

    print("Applying country-risk enrichment...")
    data = enrich_country_risk(transactions, country_risk)

    print("Creating transaction-level features...")
    data = add_transaction_features(data)

    print("Creating historical account features...")
    data = add_historical_account_features(data)

    print(f"Creating rolling network features ({lookback} lookback)...")
    data = add_rolling_network_features(data, lookback=lookback)

    print("Running AML scenarios...")
    config = ScenarioConfig()
    results = apply_scenarios(data, config)

    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_cols = [
        col for col in results.columns if col.startswith("SCN_")
    ]

    export_cols = [
        "Time",
        "Date",
        "Sender_account",
        "Receiver_account",
        "Amount",
        "Payment_currency",
        "Received_currency",
        "Sender_bank_location",
        "Receiver_bank_location",
        "Payment_type",
        "Is_laundering",
        "Laundering_type",
        "Transaction_timestamp",
        "Scenario_count",
        "Scenario_score",
        "Scenario_alert",
        "Alert_reasons",
    ] + scenario_cols

    alerts = results.loc[results["Scenario_alert"] == 1, export_cols].copy()
    alerts.to_csv(output_dir / "scenario_alerts.csv", index=False)

    summary = scenario_summary(results)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)

    if "Laundering_type" in results.columns:
        aml = results.loc[results["Is_laundering"] == 1].copy()
        coverage_rows = []
        for typology, group in aml.groupby("Laundering_type"):
            row = {
                "Laundering_type": typology,
                "AML_transactions": len(group),
            }
            for scenario in scenario_cols:
                row[scenario] = group[scenario].mean()
            row["Any_scenario_recall"] = group["Scenario_alert"].mean()
            coverage_rows.append(row)

        coverage = pd.DataFrame(coverage_rows).sort_values(
            "AML_transactions", ascending=False
        )
        coverage.to_csv(output_dir / "typology_scenario_coverage.csv", index=False)
    else:
        coverage = pd.DataFrame()

    total_alerts = int(results["Scenario_alert"].sum())
    aml_alerts = int(
        ((results["Scenario_alert"] == 1) & (results["Is_laundering"] == 1)).sum()
    )
    total_aml = int(results["Is_laundering"].sum())

    print("\n=== SCENARIO ENGINE V1 ===")
    print(f"Transactions evaluated: {len(results):,}")
    print(f"Alerts generated:       {total_alerts:,}")
    print(f"Alert rate:             {total_alerts / len(results):.2%}")
    print(f"AML transactions:       {total_aml:,}")
    print(f"AML transactions hit:   {aml_alerts:,}")
    print(
        f"Overall AML recall:     "
        f"{aml_alerts / total_aml:.2%}" if total_aml else "Overall AML recall: N/A"
    )

    print("\nScenario summary:")
    display = summary[
        [
            "Scenario",
            "Description",
            "Triggered",
            "Trigger_rate",
            "AML_cases",
            "AML_rate_when_triggered",
        ]
    ].copy()
    print(display.to_string(index=False))

    if not coverage.empty:
        print("\nRecall by laundering typology:")
        print(
            coverage[
                ["Laundering_type", "AML_transactions", "Any_scenario_recall"]
            ].to_string(index=False)
        )

    print(f"\nOutputs saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the AML scenario engine on development data."
    )
    parser.add_argument(
        "--transactions",
        type=Path,
        default=Path("data/SAML-D_sample.csv"),
    )
    parser.add_argument(
        "--country-risk",
        type=Path,
        default=Path("data/country_risk.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/scenario_run"),
    )
    parser.add_argument("--lookback", default="24h")
    args = parser.parse_args()

    run_scenarios(
        transactions_path=args.transactions,
        country_risk_path=args.country_risk,
        output_dir=args.output_dir,
        lookback=args.lookback,
    )


if __name__ == "__main__":
    main()
