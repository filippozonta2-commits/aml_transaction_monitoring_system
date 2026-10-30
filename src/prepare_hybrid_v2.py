"""Create chronological hybrid-model input files from one event-sorted source.

Build history ONCE across the entire timeline, then partition. Never recompute
rolling histories independently per split (that would erase valid past context).
This reference pipeline is for small data; large files require a streaming
implementation and an independent reproducibility audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.causal_rolling_features_v2 import build_causal_rolling_features

REQUIRED = (
    "transaction_id", "event_time", "sender", "receiver", "Amount",
    "Is_laundering", "Max_country_risk", "Any_sanctioned_country",
    "Cross_border", "Currency_mismatch",
)


def prepare(frame: pd.DataFrame, train_end: str, development_end: str):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs")
    if frame["Is_laundering"].isna().any() or not frame["Is_laundering"].isin([0, 1]).all():
        raise ValueError("Invalid AML labels")
    frame = frame.copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    if frame["event_time"].isna().any() or not frame["event_time"].is_monotonic_increasing:
        raise ValueError("Unsorted or invalid timestamps")
    first = pd.Timestamp(train_end, tz="UTC") if pd.Timestamp(train_end).tz is None else pd.Timestamp(train_end).tz_convert("UTC")
    second = pd.Timestamp(development_end, tz="UTC") if pd.Timestamp(development_end).tz is None else pd.Timestamp(development_end).tz_convert("UTC")
    if first >= second:
        raise ValueError("TRAIN end must precede DEVELOPMENT end")
    # The rolling feature builder returns one record per input row, preserving
    # the original ordering, even when timestamps are identical.
    histories = build_causal_rolling_features(frame)
    for name in histories.columns:
        if name != "transaction_id":
            if name in frame:
                raise ValueError(f"Existing historical column would be overwritten: {name}")
            frame[name] = histories[name].to_numpy()
    train = frame.loc[frame["event_time"] <= first].copy()
    dev = frame.loc[(frame["event_time"] > first) & (frame["event_time"] <= second)].copy()
    test = frame.loc[frame["event_time"] > second].copy()
    if any(part.empty for part in (train, dev, test)):
        raise ValueError("Each split must contain rows")
    if not (train["event_time"].max() < dev["event_time"].min()
            and dev["event_time"].max() < test["event_time"].min()):
        raise ValueError("Split boundaries overlap")
    return train, dev, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-end", required=True, help="UTC cutoff, inclusive")
    parser.add_argument("--development-end", required=True, help="UTC cutoff, inclusive")
    parser.add_argument("--output-dir", type=Path, default=Path("data/v2"))
    args = parser.parse_args()
    train, dev, test = prepare(pd.read_csv(args.input), args.train_end, args.development_end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in zip(("train", "development", "test"), (train, dev, test)):
        part.to_csv(args.output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(part):,} rows")


if __name__ == "__main__":
    main()
