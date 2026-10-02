"""Create leakage-aware temporal splits for SAML-D.

The split boundaries are determined from the empirical date distribution.
Rows remain in chronological, contiguous periods so transaction sequences and
network structure are preserved.
"""

from pathlib import Path
import argparse
import pandas as pd


def inspect_dates(path: Path, chunksize: int):
    daily = pd.Series(dtype="int64")
    daily_aml = pd.Series(dtype="int64")

    for n, chunk in enumerate(
        pd.read_csv(path, usecols=["Date", "Is_laundering"], chunksize=chunksize),
        start=1,
    ):
        dates = pd.to_datetime(chunk["Date"], errors="coerce").dt.normalize()
        daily = daily.add(dates.value_counts(), fill_value=0)
        daily_aml = daily_aml.add(
            dates[chunk["Is_laundering"].eq(1)].value_counts(), fill_value=0
        )
        print(f"Profiled chunk {n:,}")

    profile = pd.DataFrame({
        "Transactions": daily.astype("int64"),
        "AML_transactions": daily_aml.reindex(daily.index, fill_value=0).astype("int64"),
    }).sort_index()
    profile.index.name = "Date"
    return profile.reset_index()


def choose_boundaries(profile: pd.DataFrame, train_share=.60, dev_share=.20):
    profile = profile.sort_values("Date").copy()
    profile["Cumulative"] = profile["Transactions"].cumsum()
    total = profile["Transactions"].sum()

    train_target = total * train_share
    dev_target = total * (train_share + dev_share)

    train_end = profile.loc[
        profile["Cumulative"].ge(train_target), "Date"
    ].iloc[0]
    dev_end = profile.loc[
        profile["Cumulative"].ge(dev_target), "Date"
    ].iloc[0]
    return train_end, dev_end


def write_splits(
    path: Path,
    output_dir: Path,
    train_end: pd.Timestamp,
    dev_end: pd.Timestamp,
    chunksize: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "SAML-D_train.csv",
        "development": output_dir / "SAML-D_development.csv",
        "holdout": output_dir / "SAML-D_holdout.csv",
    }
    for p in paths.values():
        if p.exists():
            p.unlink()

    first_write = {k: True for k in paths}
    counts = {k: 0 for k in paths}
    aml_counts = {k: 0 for k in paths}

    for n, chunk in enumerate(pd.read_csv(path, chunksize=chunksize), start=1):
        dates = pd.to_datetime(chunk["Date"], errors="coerce")
        masks = {
            "train": dates.le(train_end),
            "development": dates.gt(train_end) & dates.le(dev_end),
            "holdout": dates.gt(dev_end),
        }

        for name, mask in masks.items():
            part = chunk.loc[mask]
            if part.empty:
                continue
            part.to_csv(
                paths[name],
                mode="w" if first_write[name] else "a",
                header=first_write[name],
                index=False,
            )
            first_write[name] = False
            counts[name] += len(part)
            aml_counts[name] += int(part["Is_laundering"].sum())

        print(f"Wrote chunk {n:,}")

    return paths, counts, aml_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/SAML-D.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/temporal"))
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path("results/temporal_split/date_profile.csv"),
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--train-share", type=float, default=.60)
    parser.add_argument("--development-share", type=float, default=.20)
    args = parser.parse_args()

    print("Profiling date distribution...")
    profile = inspect_dates(args.input, args.chunksize)
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(args.profile_output, index=False)

    train_end, dev_end = choose_boundaries(
        profile, args.train_share, args.development_share
    )

    print("\n=== TEMPORAL BOUNDARIES ===")
    print(f"Dataset start:   {profile['Date'].min().date()}")
    print(f"Train end:       {train_end.date()}")
    print(f"Development end: {dev_end.date()}")
    print(f"Dataset end:     {profile['Date'].max().date()}")

    print("\nWriting contiguous temporal splits...")
    paths, counts, aml_counts = write_splits(
        args.input, args.output_dir, train_end, dev_end, args.chunksize
    )

    print("\n=== TEMPORAL SPLIT SUMMARY ===")
    for name in ("train", "development", "holdout"):
        rate = aml_counts[name] / counts[name] if counts[name] else 0
        print(
            f"{name:11s} | rows: {counts[name]:,} | "
            f"AML: {aml_counts[name]:,} | AML rate: {rate:.4%}"
        )
    print("\nFiles:")
    for name, path in paths.items():
        print(f"{name:11s} -> {path}")


if __name__ == "__main__":
    main()
