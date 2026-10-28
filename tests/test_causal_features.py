"""Regression tests for causal V2.2 sender features.

Run: python3 -m unittest discover -s tests -v
"""
import unittest
import pandas as pd
from src.tune_scenarios_v22 import sender_calendar_features


def sample():
    times = ["2023-04-16 09:00:00", "2023-04-17 10:00:00",
             "2023-04-17 11:00:00", "2023-04-18 12:00:00"]
    return pd.DataFrame({
        "Transaction_timestamp": pd.to_datetime(times),
        "Sender_account": ["A"] * 4,
        "Receiver_account": ["B", "B", "C", "D"],
        "Amount": [100., 200., 300., 400.],
        "Cross_border": [0, 0, 1, 1],
        "Currency_mismatch": [0, 1, 0, 1],
        "Period": ["train", "development", "development", "development"],
    })


class CausalFeaturesTest(unittest.TestCase):
    def test_future_transactions_do_not_change_past(self):
        df = sample()
        start = pd.Timestamp("2023-04-17")
        full = sender_calendar_features(df, 21, start)
        prefix = sender_calendar_features(df.iloc[:3], 21, start)
        cols = ["tx_count", "aggregate_amount", "median_amount",
                "unique_receivers", "cross_border_rate",
                "currency_mismatch_rate", "Window_id"]
        pd.testing.assert_frame_equal(full.loc[:2, cols], prefix.loc[:2, cols])

    def test_progressive_features(self):
        out = sender_calendar_features(sample(), 21, pd.Timestamp("2023-04-17"))
        self.assertEqual(out["tx_count"].tolist(), [1, 1, 2, 3])
        self.assertEqual(out["unique_receivers"].tolist(), [1, 1, 2, 3])
        self.assertEqual(out["aggregate_amount"].tolist(), [100., 200., 500., 900.])

    def test_warmup_is_not_rolling_history(self):
        # A fixed bucket anchored to development starts a new bucket at
        # development start. This documents the limitation to fix in V3.
        out = sender_calendar_features(sample(), 21, pd.Timestamp("2023-04-17"))
        self.assertNotEqual(out.loc[0, "Window_id"], out.loc[1, "Window_id"])


if __name__ == "__main__":
    unittest.main()
