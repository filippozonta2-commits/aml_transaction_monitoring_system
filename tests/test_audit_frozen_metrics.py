"""Offline arithmetic and guardrail tests; not a rerun of frozen models."""
import csv
from pathlib import Path
import tempfile
import unittest

from src.audit_frozen_metrics import audit_summary, metrics


class FrozenMetricsAuditTest(unittest.TestCase):
    def test_confusion_matrix(self):
        m = metrics(1_898_009, 2_136, 39_409, 1_798)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]),
                         (1_798, 37_611, 338, 1_858_262))
        self.assertAlmostEqual(m["recall"], 0.8417603, places=6)

    def test_invalid_counts_rejected(self):
        with self.assertRaises(ValueError):
            metrics(100, 5, 2, 4)
        with self.assertRaises(ValueError):
            metrics(100, 5, 99, 0)

    def test_mapping_arithmetic_only(self):
        rows = [
            ["Metric", "Rule-Based V3", "XGBoost V1", "Unit"],
            ["HOLDOUT transactions", "1898009", "1898009", "transactions"],
            ["AML positives", "2136", "2136", "transactions"],
            ["Flagged transactions", "127981", "39409", "transactions"],
            ["AML hits", "1225", "1798", "transactions"],
            ["Precision", "0.009572", "0.0456241", "fraction"],
            ["Recall", "0.573502", "0.8417603", "fraction"],
            ["Alert rate", "0.067429", "0.0207633", "fraction"],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.csv"
            with path.open("w", newline="") as f:
                csv.writer(f).writerows(rows)
            result = audit_summary(path)
            self.assertEqual(result["evidence_level"], "published-summary-arithmetic-only")
            self.assertIsNone(result["pr_auc"])
            rows[4][2] = "1799"
            with path.open("w", newline="") as f:
                csv.writer(f).writerows(rows)
            with self.assertRaises(ValueError):
                audit_summary(path)


if __name__ == "__main__":
    unittest.main()
