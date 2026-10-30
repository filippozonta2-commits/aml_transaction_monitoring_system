import unittest
import pandas as pd
from src.causal_rolling_features_v2 import build_causal_rolling_features


class RollingFeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "transaction_id": [1, 2, 3, 4],
            "event_time": ["2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z",
                           "2023-01-01T01:00:00Z", "2023-01-09T00:00:00Z"],
            "sender": ["A", "A", "A", "A"],
            "receiver": ["B", "C", "B", "D"],
            "Amount": [100, 200, 300, 400],
            "Is_laundering": [1, 0, 1, 0],
        })

    def test_simultaneous_transactions_cannot_see_each_other(self):
        out = build_causal_rolling_features(self.frame)
        self.assertEqual(out.loc[0, "Sender_prior_tx_count"], 0)
        self.assertEqual(out.loc[1, "Sender_prior_tx_count"], 0)
        self.assertEqual(out.loc[2, "Sender_24h_tx_count"], 2)
        self.assertEqual(out.loc[2, "Sender_window_unique_receivers"], 2)

    def test_window_expiry_and_lifetime_count(self):
        out = build_causal_rolling_features(self.frame)
        self.assertEqual(out.loc[3, "Sender_window_tx_count"], 0)
        self.assertEqual(out.loc[3, "Sender_prior_tx_count"], 3)

    def test_label_changes_do_not_change_features(self):
        a = build_causal_rolling_features(self.frame)
        altered = self.frame.copy()
        altered["Is_laundering"] = 1 - altered["Is_laundering"]
        pd.testing.assert_frame_equal(a, build_causal_rolling_features(altered))

    def test_future_append_cannot_change_past(self):
        a = build_causal_rolling_features(self.frame)
        future = pd.DataFrame([{
            "transaction_id": 5, "event_time": "2023-01-10T00:00:00Z",
            "sender": "A", "receiver": "B", "Amount": 999, "Is_laundering": 1,
        }])
        b = build_causal_rolling_features(pd.concat([self.frame, future], ignore_index=True))
        pd.testing.assert_frame_equal(a, b.iloc[:len(a)].reset_index(drop=True))

    def test_duplicate_ids_rejected(self):
        altered = self.frame.copy()
        altered.loc[1, "transaction_id"] = 1
        with self.assertRaises(ValueError):
            build_causal_rolling_features(altered)


if __name__ == "__main__":
    unittest.main()
