import unittest
import pandas as pd
from src.prepare_hybrid_v2 import prepare


class PrepareHybridTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "transaction_id": [1, 2, 3, 4],
            "event_time": ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z",
                           "2023-01-03T00:00:00Z", "2023-01-04T00:00:00Z"],
            "sender": ["A", "A", "A", "A"], "receiver": ["B", "C", "D", "E"],
            "Amount": [10., 20., 30., 40.],
            "Is_laundering": [0, 1, 0, 1],
            "Max_country_risk": [1., 1., 1., 1.],
            "Any_sanctioned_country": [0, 0, 0, 0],
            "Cross_border": [0, 0, 0, 0],
            "Currency_mismatch": [0, 0, 0, 0],
        })

    def test_history_continues_across_split_boundaries(self):
        train, dev, test = prepare(self.frame, "2023-01-02", "2023-01-03")
        self.assertEqual(dev.iloc[0]["Sender_prior_tx_count"], 2)
        self.assertEqual(test.iloc[0]["Sender_prior_tx_count"], 3)
        self.assertEqual(test.iloc[0]["Sender_window_unique_receivers"], 3)

    def test_unsorted_input_rejected(self):
        shuffled = self.frame.iloc[[1, 0, 2, 3]].reset_index(drop=True)
        with self.assertRaises(ValueError):
            prepare(shuffled, "2023-01-02", "2023-01-03")

    def test_invalid_boundaries_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.frame, "2023-01-03", "2023-01-02")


if __name__ == "__main__":
    unittest.main()
