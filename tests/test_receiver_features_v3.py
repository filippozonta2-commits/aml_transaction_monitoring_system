import unittest
import pandas as pd
from src.receiver_features_v3 import receiver_rolling_features


class ReceiverFeaturesV3Test(unittest.TestCase):
    def test_warmup_and_expiration_and_isolation(self):
        df = pd.DataFrame({
            "Transaction_timestamp": pd.to_datetime([
                "2026-01-01", "2026-01-02", "2026-01-12", "2026-01-12"
            ]),
            "Receiver_account": ["A", "A", "A", "B"],
            "Sender_account": ["X", "Y", "Y", "Z"],
            "Amount": [100., 200., 300., 400.],
            "Cross_border": [0, 1, 0, 1],
            "Currency_mismatch": [0, 0, 1, 1],
        })
        result = receiver_rolling_features(df, 10)
        self.assertEqual(result["receiver_tx_count"].tolist(), [1, 2, 2, 1])
        self.assertEqual(result["receiver_unique_senders"].tolist(), [1, 2, 1, 1])
        self.assertEqual(result["receiver_aggregate_amount"].tolist(), [100., 300., 500., 400.])

    def test_future_does_not_change_past(self):
        df = pd.DataFrame({
            "Transaction_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Receiver_account": ["A", "A"],
            "Sender_account": ["X", "Y"],
            "Amount": [100., 200.],
            "Cross_border": [0, 1],
            "Currency_mismatch": [0, 1],
        })
        one = receiver_rolling_features(df.iloc[:1], 10)
        two = receiver_rolling_features(df, 10)
        for col in ["receiver_tx_count", "receiver_aggregate_amount",
                    "receiver_unique_senders", "receiver_cross_border_rate",
                    "receiver_currency_mismatch_rate"]:
            self.assertEqual(one[col].iloc[0], two[col].iloc[0])


if __name__ == "__main__":
    unittest.main()
