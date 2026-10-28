"""Unit tests for V3 rolling sender features."""
import unittest
import pandas as pd
from src.scenario_features_v3 import sender_rolling_features


def example():
    return pd.DataFrame({
        "Transaction_timestamp": pd.to_datetime([
            "2023-04-15 12:00", "2023-04-17 12:00",
            "2023-04-18 12:00", "2023-04-25 12:00",
        ]),
        "Sender_account": ["A", "A", "A", "A"],
        "Receiver_account": ["B", "B", "C", "D"],
        "Amount": [100., 200., 300., 400.],
        "Cross_border": [0, 1, 1, 0],
        "Currency_mismatch": [0, 0, 1, 0],
        "Period": ["train", "development", "development", "development"],
    })


class RollingFeaturesTest(unittest.TestCase):
    def test_train_history_contributes_to_development(self):
        result = sender_rolling_features(example(), 7)
        self.assertEqual(result["tx_count"].tolist()[:3], [1, 2, 3])
        self.assertEqual(result["aggregate_amount"].tolist()[:3], [100., 300., 600.])
        self.assertEqual(result["median_amount"].tolist()[:3], [100., 150., 200.])

    def test_expired_events_removed(self):
        result = sender_rolling_features(example(), 7)
        self.assertEqual(result["tx_count"].iloc[-1], 1)
        self.assertEqual(result["aggregate_amount"].iloc[-1], 400.)
        self.assertEqual(result["unique_receivers"].iloc[-1], 1)

    def test_future_does_not_change_prefix(self):
        data = example()
        full = sender_rolling_features(data, 7)
        prefix = sender_rolling_features(data.iloc[:3], 7)
        columns = ["tx_count", "aggregate_amount", "median_amount",
                   "unique_receivers", "cross_border_rate",
                   "currency_mismatch_rate"]
        pd.testing.assert_frame_equal(full.loc[:2, columns], prefix.loc[:2, columns])

    def test_senders_independent(self):
        data = example()
        data.loc[2, "Sender_account"] = "Z"
        result = sender_rolling_features(data, 7)
        self.assertEqual(result["tx_count"].tolist()[:3], [1, 2, 1])


if __name__ == "__main__":
    unittest.main()
