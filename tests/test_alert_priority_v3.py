import unittest
import pandas as pd
from src.alert_priority_v3 import prioritize_alerts


class AlertPriorityTest(unittest.TestCase):
    def test_explainability_and_deduplication(self):
        tx = pd.DataFrame({"Amount": [100, 200, 300], "Cross_border": [0, 1, 0],
                           "Currency_mismatch": [0, 1, 0]})
        flags = pd.DataFrame({"Smurfing": [True, False, False],
                              "Fan_Out": [True, True, False],
                              "Structuring": [False, False, False],
                              "Fan_In": [False, False, False]})
        alerts = prioritize_alerts(tx, flags)
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts.loc[0, "Scenario_count"], 2)
        self.assertEqual(alerts.loc[0, "Priority_score"], 55)
        self.assertEqual(alerts.loc[1, "Priority_score"], 60)
        self.assertIn("Fan_Out", alerts.loc[0, "Triggered_scenarios"])

    def test_rejects_misaligned_indices(self):
        tx = pd.DataFrame({"Amount": [100]}, index=[1])
        flags = pd.DataFrame({"Smurfing": [True], "Fan_Out": [False],
                              "Structuring": [False], "Fan_In": [False]}, index=[2])
        with self.assertRaises(ValueError):
            prioritize_alerts(tx, flags)


if __name__ == "__main__":
    unittest.main()
