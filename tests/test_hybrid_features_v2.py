import unittest
import pandas as pd
from src.hybrid_features_v2 import build_hybrid_features


class HybridFeatureTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "Amount": [9800., 1200.], "Max_country_risk": [3., 8.],
            "Any_sanctioned_country": [0, 0],
            "Cross_border": [1, 0], "Currency_mismatch": [1, 0],
            "Sender_seconds_since_previous": [120., 100000.],
            "Sender_prior_tx_count": [12, 1],
            "Sender_prior_avg_amount": [2000., 1500.],
            "Is_laundering": [1, 0],
        })

    def test_label_change_cannot_change_features(self):
        a = build_hybrid_features(self.data)
        changed = self.data.copy()
        changed["Is_laundering"] = [0, 1]
        pd.testing.assert_frame_equal(a, build_hybrid_features(changed))
        self.assertNotIn("Is_laundering", a.columns)
        self.assertNotIn("Scenario_alert", a.columns)

    def test_partial_network_history_is_rejected(self):
        changed = self.data.copy()
        changed["Sender_window_tx_count"] = [1, 2]
        with self.assertRaises(ValueError):
            build_hybrid_features(changed)

    def test_missing_required_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            build_hybrid_features(self.data.drop(columns=["Sender_prior_tx_count"]))


if __name__ == "__main__":
    unittest.main()
