"""Read-only schema and temporal readiness audit for a local SAML-D CSV.

Does not infer columns, compute target-derived country risk, or touch HOLDOUT
labels for feature engineering. Outputs a JSON report; never exports raw rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CANONICAL = {
    "transaction_id": ("transaction_id", "Transaction_id", "Transaction_ID"),
    "event_time": ("event_time", "Date", "Timestamp"),
    "sender": ("sender", "Sender_account", "Sender_id"),
    "receiver": ("receiver", "Receiver_account", "Receiver_id"),
    "Amount": ("Amount", "Amount_paid"),
    "Is_laundering": ("Is_laundering",),
    "Max_country_risk": ("Max_country_risk",),
    "Any_sanctioned_country": ("Any_sanctioned_country",),
    "Cross_border": ("Cross_border",),
    "Currency_mismatch": ("Currency_mismatch",),
}
PROVENANCE_REQUIRED = {
    "Max_country_risk": "External country-risk source and effective date",
    "Any_sanctioned_country": "Sanctions list effective at event time",
    "Cross_border": "Sender/receiver country fields",
    "Currency_mismatch": "Transaction and account currency fields",
}


def audit_schema(columns):
    present = set(columns)
    matches = {key: [name for name in options if name in present]
               for key, options in CANONICAL.items()}
    return {
        "candidate_matches": matches,
        "unresolved": [key for key, found in matches.items() if not found],
        "ambiguous": [key for key, found in matches.items() if len(found) > 1],
        "requires_external_provenance": PROVENANCE_REQUIRED,
        "ready_for_training": False,
        "note": "Column presence alone never establishes causal provenance or readiness.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/hybrid_v2/schema_audit.json"))
    args = parser.parse_args()
    cols = list(pd.read_csv(args.input, nrows=0).columns)
    report = audit_schema(cols)
    report["input_filename"] = args.input.name
    report["column_count"] = len(cols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
