"""Memory-efficient initial EDA and development sampling for SAML-D."""

from pathlib import Path
import argparse

import pandas as pd


TARGET = "Is_laundering"
TYPOLOGY = "Laundering_type"


def profile_and_sample(
    input_path: Path,
    sample_path: Path,
    summary_dir: Path,
    chunksize: int = 250_000,
    normal_sample_per_chunk: int = 5_000,
    aml_sample_per_chunk: int = 5_000,
    random_state: int = 42,
) -> None:
    """Profile SAML-D in chunks and create a class-aware development sample."""
    total_rows = 0
    class_counts = pd.Series(dtype="int64")
    typology_counts = pd.Series(dtype="int64")
    payment_type_counts = pd.Series(dtype="int64")
    sender_country_counts = pd.Series(dtype="int64")
    receiver_country_counts = pd.Series(dtype="int64")
    currency_counts = pd.Series(dtype="int64")
    sampled_chunks = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(input_path, chunksize=chunksize), start=1
    ):
        total_rows += len(chunk)

        class_counts = class_counts.add(
            chunk[TARGET].value_counts(), fill_value=0
        )

        aml = chunk[chunk[TARGET] == 1]
        typology_counts = typology_counts.add(
            aml[TYPOLOGY].value_counts(), fill_value=0
        )

        payment_type_counts = payment_type_counts.add(
            chunk["Payment_type"].value_counts(), fill_value=0
        )
        sender_country_counts = sender_country_counts.add(
            chunk["Sender_bank_location"].value_counts(), fill_value=0
        )
        receiver_country_counts = receiver_country_counts.add(
            chunk["Receiver_bank_location"].value_counts(), fill_value=0
        )
        currency_counts = currency_counts.add(
            chunk["Payment_currency"].value_counts(), fill_value=0
        )

        normal = chunk[chunk[TARGET] == 0]

        if len(normal):
            sampled_chunks.append(
                normal.sample(
                    n=min(normal_sample_per_chunk, len(normal)),
                    random_state=random_state + chunk_number,
                )
            )

        if len(aml):
            sampled_chunks.append(
                aml.sample(
                    n=min(aml_sample_per_chunk, len(aml)),
                    random_state=random_state + 10_000 + chunk_number,
                )
            )

        print(
            f"Processed chunk {chunk_number:,} | "
            f"rows so far: {total_rows:,} | "
            f"AML in chunk: {len(aml):,}"
        )

    summary_dir.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    class_counts = class_counts.astype("int64").sort_index()
    typology_counts = typology_counts.astype("int64").sort_values(ascending=False)

    class_summary = class_counts.rename_axis(TARGET).reset_index(name="Transactions")
    class_summary["Percentage"] = (
        class_summary["Transactions"] / class_summary["Transactions"].sum() * 100
    )

    typology_summary = typology_counts.rename_axis(TYPOLOGY).reset_index(
        name="Transactions"
    )
    if len(typology_summary):
        typology_summary["Percentage_of_AML"] = (
            typology_summary["Transactions"]
            / typology_summary["Transactions"].sum()
            * 100
        )

    class_summary.to_csv(summary_dir / "class_distribution.csv", index=False)
    typology_summary.to_csv(summary_dir / "laundering_typologies.csv", index=False)

    payment_type_counts.sort_values(ascending=False).rename("Transactions").to_csv(
        summary_dir / "payment_types.csv"
    )
    sender_country_counts.sort_values(ascending=False).rename("Transactions").to_csv(
        summary_dir / "sender_countries.csv"
    )
    receiver_country_counts.sort_values(ascending=False).rename("Transactions").to_csv(
        summary_dir / "receiver_countries.csv"
    )
    currency_counts.sort_values(ascending=False).rename("Transactions").to_csv(
        summary_dir / "payment_currencies.csv"
    )

    sample = pd.concat(sampled_chunks, ignore_index=True)
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    sample.to_csv(sample_path, index=False)

    print("\n=== SAML-D INITIAL PROFILE ===")
    print(f"Total rows: {total_rows:,}")
    print("\nClass distribution:")
    print(class_summary.to_string(index=False))
    print("\nAML typologies:")
    print(typology_summary.to_string(index=False))
    print(f"\nDevelopment sample: {len(sample):,} rows")
    print(f"Saved to: {sample_path}")
    print(f"Summaries saved to: {summary_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunked EDA and class-aware sampling for SAML-D."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/SAML-D.csv"),
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("data/SAML-D_sample.csv"),
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("results/eda"),
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--normal-per-chunk", type=int, default=5_000)
    parser.add_argument("--aml-per-chunk", type=int, default=5_000)
    args = parser.parse_args()

    profile_and_sample(
        input_path=args.input,
        sample_path=args.sample_output,
        summary_dir=args.summary_dir,
        chunksize=args.chunksize,
        normal_sample_per_chunk=args.normal_per_chunk,
        aml_sample_per_chunk=args.aml_per_chunk,
    )


if __name__ == "__main__":
    main()
