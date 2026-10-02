"""Chunked AML typology profiling for the full SAML-D dataset."""

from pathlib import Path
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


TARGET = "Is_laundering"
TYPOLOGY = "Laundering_type"

COUNTRY_MAP = {
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "USA": "United States",
}


def _nested_counter():
    return defaultdict(lambda: defaultdict(int))


def profile_typologies(
    input_path: Path,
    output_dir: Path,
    chunksize: int = 250_000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    amount_sum = defaultdict(float)
    amount_sum_sq = defaultdict(float)
    amount_min = {}
    amount_max = {}

    cross_border = defaultdict(int)
    currency_mismatch = defaultdict(int)
    night = defaultdict(int)
    weekend = defaultdict(int)

    payment_types = _nested_counter()
    payment_currencies = _nested_counter()
    received_currencies = _nested_counter()
    sender_countries = _nested_counter()
    receiver_countries = _nested_counter()

    aml_rows = []

    usecols = [
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
        TARGET,
        TYPOLOGY,
    ]

    for chunk_number, chunk in enumerate(
        pd.read_csv(input_path, usecols=usecols, chunksize=chunksize), start=1
    ):
        aml = chunk.loc[chunk[TARGET] == 1].copy()
        if aml.empty:
            continue

        aml["Sender_bank_location"] = aml["Sender_bank_location"].replace(COUNTRY_MAP)
        aml["Receiver_bank_location"] = aml["Receiver_bank_location"].replace(COUNTRY_MAP)

        aml["Cross_border"] = (
            aml["Sender_bank_location"] != aml["Receiver_bank_location"]
        ).astype(int)
        aml["Currency_mismatch"] = (
            aml["Payment_currency"] != aml["Received_currency"]
        ).astype(int)

        dates = pd.to_datetime(aml["Date"], errors="coerce")
        times = pd.to_datetime(aml["Time"], format="%H:%M:%S", errors="coerce")
        aml["Is_weekend"] = dates.dt.dayofweek.ge(5).astype(int)
        aml["Is_night"] = times.dt.hour.between(0, 5).astype(int)

        # Only AML rows are retained; there are ~10k, so this remains lightweight.
        aml_rows.append(aml)

        for typology, group in aml.groupby(TYPOLOGY):
            n = len(group)
            amounts = group["Amount"].astype(float)

            counts[typology] += n
            amount_sum[typology] += amounts.sum()
            amount_sum_sq[typology] += np.square(amounts).sum()
            amount_min[typology] = min(
                amount_min.get(typology, float("inf")), amounts.min()
            )
            amount_max[typology] = max(
                amount_max.get(typology, float("-inf")), amounts.max()
            )

            cross_border[typology] += int(group["Cross_border"].sum())
            currency_mismatch[typology] += int(group["Currency_mismatch"].sum())
            night[typology] += int(group["Is_night"].sum())
            weekend[typology] += int(group["Is_weekend"].sum())

            for value, value_count in group["Payment_type"].value_counts().items():
                payment_types[typology][str(value)] += int(value_count)
            for value, value_count in group["Payment_currency"].value_counts().items():
                payment_currencies[typology][str(value)] += int(value_count)
            for value, value_count in group["Received_currency"].value_counts().items():
                received_currencies[typology][str(value)] += int(value_count)
            for value, value_count in group["Sender_bank_location"].value_counts().items():
                sender_countries[typology][str(value)] += int(value_count)
            for value, value_count in group["Receiver_bank_location"].value_counts().items():
                receiver_countries[typology][str(value)] += int(value_count)

        print(
            f"Processed chunk {chunk_number:,} | "
            f"AML rows in chunk: {len(aml):,}"
        )

    aml_all = pd.concat(aml_rows, ignore_index=True)

    # Exact quantiles are inexpensive because the AML population is only ~10k rows.
    quantiles = (
        aml_all.groupby(TYPOLOGY)["Amount"]
        .quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .unstack()
    )

    rows = []
    for typology in sorted(counts):
        n = counts[typology]
        mean = amount_sum[typology] / n
        variance = max(amount_sum_sq[typology] / n - mean**2, 0)

        rows.append(
            {
                "Laundering_type": typology,
                "Transactions": n,
                "Percentage_of_AML": n / len(aml_all) * 100,
                "Amount_mean": mean,
                "Amount_std": variance**0.5,
                "Amount_min": amount_min[typology],
                "Amount_p25": quantiles.loc[typology, 0.25],
                "Amount_median": quantiles.loc[typology, 0.50],
                "Amount_p75": quantiles.loc[typology, 0.75],
                "Amount_p90": quantiles.loc[typology, 0.90],
                "Amount_p95": quantiles.loc[typology, 0.95],
                "Amount_p99": quantiles.loc[typology, 0.99],
                "Amount_max": amount_max[typology],
                "Cross_border_rate": cross_border[typology] / n,
                "Currency_mismatch_rate": currency_mismatch[typology] / n,
                "Night_rate": night[typology] / n,
                "Weekend_rate": weekend[typology] / n,
                "Unique_senders": aml_all.loc[
                    aml_all[TYPOLOGY] == typology, "Sender_account"
                ].nunique(),
                "Unique_receivers": aml_all.loc[
                    aml_all[TYPOLOGY] == typology, "Receiver_account"
                ].nunique(),
            }
        )

    summary = pd.DataFrame(rows).sort_values("Transactions", ascending=False)
    summary.to_csv(output_dir / "typology_profile.csv", index=False)

    def save_long(counter, filename, dimension):
        records = []
        for typology, values in counter.items():
            total = sum(values.values())
            for value, count in sorted(
                values.items(), key=lambda item: item[1], reverse=True
            ):
                records.append(
                    {
                        "Laundering_type": typology,
                        dimension: value,
                        "Transactions": count,
                        "Share_within_typology": count / total if total else 0,
                    }
                )
        pd.DataFrame(records).to_csv(output_dir / filename, index=False)

    save_long(payment_types, "typology_payment_types.csv", "Payment_type")
    save_long(payment_currencies, "typology_payment_currencies.csv", "Payment_currency")
    save_long(received_currencies, "typology_received_currencies.csv", "Received_currency")
    save_long(sender_countries, "typology_sender_countries.csv", "Sender_country")
    save_long(receiver_countries, "typology_receiver_countries.csv", "Receiver_country")

    # Keep a compact AML-only file locally for network/sequence exploration.
    aml_all.to_csv(output_dir / "aml_transactions.csv", index=False)

    print("\n=== TYPOLOGY PROFILE ===")
    display_cols = [
        "Laundering_type",
        "Transactions",
        "Amount_median",
        "Amount_p95",
        "Cross_border_rate",
        "Currency_mismatch_rate",
        "Unique_senders",
        "Unique_receivers",
    ]
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved detailed outputs to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile SAML-D AML typologies using chunked full-data reads."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/SAML-D.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/typology_eda")
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    profile_typologies(args.input, args.output_dir, args.chunksize)


if __name__ == "__main__":
    main()
