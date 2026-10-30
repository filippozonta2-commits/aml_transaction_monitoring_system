"""Read-only provenance gate for the original governed country-risk CSV.

This does not derive sanctions or change the frozen V3 thresholds. In particular,
the historical materializer interprets ANY nonempty 'Sanction' cell as true.
Textual negatives ('No', 'False', '0') would therefore be misclassified.
Fail closed until the original reference semantics are documented.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REQUIRED = ("Country", "ISO Code", "Overall Score", "ML/TF Risk", "FATF", "Sanction")
NEGATIVE_TEXT = frozenset({
    "no", "n", "false", "0", "none", "not sanctioned",
    "no sanctions", "not applicable", "n/a", "unsanctioned",
})


def audit(path: Path) -> dict:
    problems: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = sorted(set(REQUIRED) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required country-risk columns: {missing}")
        seen: set[str] = set()
        count = 0
        ambiguous = 0
        for line, row in enumerate(reader, start=2):
            count += 1
            iso = (row["ISO Code"] or "").strip().upper()
            if len(iso) != 2 or not iso.isalpha():
                problems.append(f"row {line}: invalid ISO-2 code {iso!r}")
            if iso in seen:
                problems.append(f"row {line}: duplicate ISO-2 code {iso!r}")
            seen.add(iso)
            try:
                score = float((row["Overall Score"] or "").strip())
                if not math.isfinite(score) or score < 0:
                    raise ValueError()
            except ValueError:
                problems.append(f"row {line}: invalid Overall Score")
            sanctions = (row["Sanction"] or "").strip().lower()
            if sanctions in NEGATIVE_TEXT:
                ambiguous += 1
                problems.append(
                    f"row {line}: textual negative Sanction={sanctions!r} "
                    "would be treated as sanctioned by the frozen nonempty-cell rule"
                )
        if count == 0:
            problems.append("Empty country-risk registry")
    return {
        "rows": count,
        "ambiguous_negative_sanction_rows": ambiguous,
        "passed": not problems,
        "problems": problems,
        "evidence_level": "reference-schema-and-semantics-only",
        "note": "Passing does not establish source provenance or event-time validity.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country-risk", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.country_risk)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
