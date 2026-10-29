"""Tests for frozen V3 preflight, independent of original dataset."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.preflight_frozen_v3 import REQUIRED_RISK, REQUIRED_TX, header, main


class PreflightFrozenV3Tests(unittest.TestCase):
    def write_csv(self, path, cols):
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(sorted(cols))

    def test_headers(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "risk.csv"
            self.write_csv(p, REQUIRED_RISK)
            self.assertEqual(header(p), REQUIRED_RISK)

    def test_missing_country_risk_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / "source.csv"
            self.write_csv(src, REQUIRED_TX)
            args = ["preflight", "--source", str(src),
                    "--country-risk", str(root / "missing.csv"),
                    "--train", str(src), "--development", str(src),
                    "--holdout", str(src)]
            with patch("sys.argv", args):
                self.assertEqual(main(), 2)


if __name__ == "__main__":
    unittest.main()
