"""Synthetic temporal-split contract tests; no raw SAML-D data required."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.create_temporal_split import inspect_dates, choose_boundaries, write_splits


class TemporalSplitContractTest(unittest.TestCase):
    def test_invalid_date_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "tx.csv"
            pd.DataFrame({
                "Date": ["2023-01-01", "not-a-date", "2023-01-03"],
                "Is_laundering": [0, 1, 0],
            }).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "Unparseable Date"):
                inspect_dates(source, chunksize=2)

    def test_invalid_split_shares_rejected(self):
        profile = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=5),
            "Transactions": [1] * 5,
        })
        with self.assertRaises(ValueError):
            choose_boundaries(profile, train_share=.8, dev_share=.3)

    def test_full_partition_and_date_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "tx.csv"
            rows = pd.DataFrame({
                "Date": [f"2023-01-0{i} 12:00:00" for i in range(1, 6)],
                "Time": ["12:00:00"] * 5,
                "Is_laundering": [0, 1, 0, 0, 1],
            })
            rows.to_csv(source, index=False)
            profile = inspect_dates(source, chunksize=2)
            train_end, dev_end = choose_boundaries(profile, .6, .2)
            _, counts, positives = write_splits(
                source, Path(tmp) / "splits", train_end, dev_end, chunksize=2
            )
            self.assertEqual(counts, {"train": 3, "development": 1, "holdout": 1})
            self.assertEqual(sum(counts.values()), len(rows))
            self.assertEqual(sum(positives.values()), int(rows.Is_laundering.sum()))


if __name__ == "__main__":
    unittest.main()
