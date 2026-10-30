"""Offline fixtures for the governed country-risk reference gate."""
import csv
import tempfile
import unittest
from pathlib import Path

from src.audit_country_risk_reference import REQUIRED, audit


class CountryRiskReferenceAuditTests(unittest.TestCase):
    def check(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "risk.csv"
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=REQUIRED)
                w.writeheader()
                for row in rows:
                    w.writerow({
                        "Country": "Example", "ISO Code": "IT",
                        "Overall Score": "6.5", "ML/TF Risk": "",
                        "FATF": "", "Sanction": "", **row,
                    })
            return audit(p)

    def test_empty_and_named_sanction_are_unambiguous(self):
        r = self.check([
            {"ISO Code": "IT", "Sanction": ""},
            {"ISO Code": "US", "Sanction": "Named restriction"},
        ])
        self.assertTrue(r["passed"])
        self.assertEqual(r["rows"], 2)

    def test_textual_no_must_not_be_silently_treated_as_sanction(self):
        r = self.check([{"Sanction": "No"}])
        self.assertFalse(r["passed"])
        self.assertEqual(r["ambiguous_negative_sanction_rows"], 1)

    def test_duplicate_iso_is_rejected(self):
        r = self.check([{}, {}])
        self.assertFalse(r["passed"])
        self.assertTrue(any("duplicate" in p for p in r["problems"]))

    def test_non_numeric_risk_score_is_rejected(self):
        r = self.check([{"Overall Score": "unknown"}])
        self.assertFalse(r["passed"])


if __name__ == "__main__":
    unittest.main()
