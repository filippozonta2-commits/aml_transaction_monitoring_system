import unittest
import numpy as np
import pandas as pd
from src.train_hybrid_v2 import check_splits, choose_threshold, report, reject_unaudited_columns


class HybridExperimentTests(unittest.TestCase):
    def _frame(self, ids, times):
        return pd.DataFrame({
            "transaction_id": ids, "event_time": pd.to_datetime(times, utc=True),
            "Is_laundering": [0, 1],
        })

    def test_disjoint_temporal_splits(self):
        a = self._frame([1, 2], ["2023-01-01", "2023-01-02"])
        b = self._frame([3, 4], ["2023-02-01", "2023-02-02"])
        c = self._frame([5, 6], ["2023-03-01", "2023-03-02"])
        check_splits(a, b, c)
        c.loc[0, "transaction_id"] = 2
        with self.assertRaises(ValueError):
            check_splits(a, b, c)

    def test_overlap_time_rejected(self):
        a = self._frame([1, 2], ["2023-01-01", "2023-01-02"])
        b = self._frame([3, 4], ["2023-01-02", "2023-01-03"])
        c = self._frame([5, 6], ["2023-03-01", "2023-03-02"])
        with self.assertRaises(ValueError):
            check_splits(a, b, c)

    def test_budget_respected_even_with_ties(self):
        scores = np.array([0.9, 0.8, 0.8, 0.1, 0.0])
        cutoff = choose_threshold(np.array([1, 0, 1, 0, 0]), scores, 0.4)
        self.assertLessEqual(int((scores >= cutoff).sum()), 2)

    def test_unique_scores_fill_alert_capacity(self):
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
        cutoff = choose_threshold(np.array([1, 0, 1, 0, 0]), scores, 0.4)
        self.assertEqual(int((scores >= cutoff).sum()), 2)

    def test_rejects_country_target_rate_even_if_not_selected(self):
        clean = pd.DataFrame({"transaction_id": [1]})
        tainted = clean.assign(country_aml_rate=[0.4])
        with self.assertRaises(ValueError):
            reject_unaudited_columns(clean, clean, tainted)

    def test_rejects_post_decision_alert(self):
        clean = pd.DataFrame({"transaction_id": [1]})
        with self.assertRaises(ValueError):
            reject_unaudited_columns(clean.assign(Scenario_alert=[1]), clean, clean)

    def test_metrics_are_derived_from_rows(self):
        r = report(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.1, 0.0]), 0.5)
        self.assertEqual(r["true_positives"], 1)
        self.assertEqual(r["flagged"], 2)
        self.assertEqual(r["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
