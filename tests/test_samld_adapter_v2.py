import unittest
import pandas as pd
from src.samld_adapter_v2 import adapt_samld


class SamldAdapterTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "Date": ["2022-10-07", "2022-10-07"],
            "Time": ["10:35:19", "10:35:20"],
            "Sender_account": ["8724731955", "1491989064"],
            "Receiver_account": ["2769355426", "8401255335"],
            "Amount": [1459.15, 6019.64],
            "Payment_currency": ["UK pounds", "UK pounds"],
            "Received_currency": ["UK pounds", "Dirham"],
            "Sender_bank_location": ["UK", "UK"],
            "Receiver_bank_location": ["UK", "UAE"],
            "Is_laundering": [0, 1],
            "Laundering_type": ["Normal_Cash_Deposits", "Normal_Fan_Out"],
        })

    def test_original_columns_map_without_derived_risk(self):
        out = adapt_samld(self.raw)
        self.assertEqual(out["Cross_border"].tolist(), [0, 1])
        self.assertEqual(out["Currency_mismatch"].tolist(), [0, 1])
        self.assertEqual(out["transaction_id"].tolist(), [0, 1])
        self.assertNotIn("Max_country_risk", out)
        self.assertNotIn("Any_sanctioned_country", out)
        self.assertNotIn("Laundering_type", out)

    def test_labels_cannot_change_features(self):
        a = adapt_samld(self.raw).drop(columns=["Is_laundering"])
        altered = self.raw.copy()
        altered["Is_laundering"] = [1, 0]
        altered["Laundering_type"] = ["fake", "fake"]
        b = adapt_samld(altered).drop(columns=["Is_laundering"])
        pd.testing.assert_frame_equal(a, b)

    def test_row_offset_preserves_unique_ids_across_chunks(self):
        a = adapt_samld(self.raw.iloc[:1], start_row=0)
        b = adapt_samld(self.raw.iloc[1:], start_row=1)
        self.assertEqual(a["transaction_id"].tolist() + b["transaction_id"].tolist(), [0, 1])

    def test_missing_input_fails_closed(self):
        with self.assertRaises(ValueError):
            adapt_samld(self.raw.drop(columns=["Time"]))


if __name__ == "__main__":
    unittest.main()
