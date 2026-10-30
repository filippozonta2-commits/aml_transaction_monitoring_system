import unittest
from src.audit_hybrid_schema_v2 import audit_schema


class SchemaAuditTests(unittest.TestCase):
    def test_missing_derived_risk_fields_are_reported(self):
        report = audit_schema(["Date", "Is_laundering", "Amount"])
        self.assertIn("Max_country_risk", report["unresolved"])
        self.assertFalse(report["ready_for_training"])

    def test_ambiguous_aliases_are_not_silently_chosen(self):
        report = audit_schema(["Date", "event_time"])
        self.assertIn("event_time", report["ambiguous"])

    def test_all_fields_still_require_provenance(self):
        report = audit_schema([
            "transaction_id", "event_time", "sender", "receiver",
            "Amount", "Is_laundering", "Max_country_risk",
            "Any_sanctioned_country", "Cross_border", "Currency_mismatch",
        ])
        self.assertEqual(report["unresolved"], [])
        self.assertFalse(report["ready_for_training"])


if __name__ == "__main__":
    unittest.main()
