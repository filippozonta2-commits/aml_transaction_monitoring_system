"""Preflight original frozen 11-control V3 inputs without accessing HOLDOUT labels."""
from pathlib import Path
import argparse
import csv
import importlib.util
import sys
import zipfile

REQUIRED_RISK = {"Country", "ISO Code", "Overall Score", "ML/TF Risk", "FATF", "Sanction"}
REQUIRED_TX = {"Date", "Time", "Sender_account", "Receiver_account", "Amount",
               "Payment_type", "Payment_currency", "Received_currency",
               "Sender_bank_location", "Receiver_bank_location",
               "Is_laundering", "Laundering_type"}


def header(path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            files = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if len(files) != 1:
                raise ValueError("ZIP must contain exactly one CSV")
            with z.open(files[0]) as f:
                return set(f.readline().decode("utf-8-sig").strip().split(","))
    with path.open(encoding="utf-8-sig", newline="") as f:
        return set(next(csv.reader(f)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True,
                   help="Original SAML-D.csv or ZIP containing it")
    p.add_argument("--country-risk", type=Path, default=Path("data/country_risk.csv"))
    p.add_argument("--train", type=Path, default=Path("data/temporal/SAML-D_train.csv"))
    p.add_argument("--development", type=Path, default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--holdout", type=Path, default=Path("data/temporal/SAML-D_holdout.csv"))
    a = p.parse_args()
    issues = []
    if not a.source.is_file():
        issues.append(f"Missing original source: {a.source}")
    else:
        missing = REQUIRED_TX - header(a.source)
        if missing:
            issues.append(f"Original source missing columns: {sorted(missing)}")
        else:
            print("OK: original transaction source schema")
    if not a.country_risk.is_file():
        issues.append(f"Missing ORIGINAL country-risk reference: {a.country_risk}")
    else:
        missing = REQUIRED_RISK - header(a.country_risk)
        if missing:
            issues.append(f"Country-risk reference missing columns: {sorted(missing)}")
        else:
            print("OK: original country-risk reference schema")
    for path in (a.train, a.development, a.holdout):
        if not path.is_file():
            issues.append(f"Missing chronological split: {path}")
        else:
            missing = REQUIRED_TX - header(path)
            if missing:
                issues.append(f"Split {path} missing columns: {sorted(missing)}")
            else:
                print(f"OK: {path}")
    if importlib.util.find_spec("pyspark") is None:
        issues.append("PySpark not installed (required for frozen geography and Deposit-Send)")
    else:
        print("OK: PySpark importable")
    for issue in issues:
        print("BLOCKER:", issue)
    if issues:
        print(f"NOT READY: {len(issues)} blockers; original V3 was NOT evaluated")
        return 2
    print("PREFLIGHT PASSED: inputs available; no model evaluation performed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
