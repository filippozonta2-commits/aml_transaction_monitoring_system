"""Fail-closed verification of the *original* frozen 11-scenario V3 contract.

The generic src.rule_based engine is NOT the frozen V3 implementation. This
guard prevents a 12-flag engine or a four-flag prototype being silently
presented as a reproduction of the historical benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FIELDS = (
    "scenario_id", "implementation_path", "implementation_sha256",
    "parameters", "lookback", "entity_key",
)


def verify_manifest(manifest: dict, root: Path) -> dict:
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 11:
        raise ValueError("Frozen V3 requires exactly 11 documented scenarios")
    ids = [s.get("scenario_id") for s in scenarios]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != 11:
        raise ValueError("Scenario IDs must be 11 unique nonempty strings")
    for scenario in scenarios:
        missing = [field for field in REQUIRED_FIELDS if field not in scenario]
        if missing:
            raise ValueError(f"{scenario['scenario_id']}: missing {missing}")
        rel = Path(scenario["implementation_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Implementation must be a safe repository-relative path")
        source = (root / rel).resolve()
        if not source.is_file() or not source.is_relative_to(root.resolve()):
            raise ValueError(f"Missing scenario source: {rel}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != scenario["implementation_sha256"]:
            raise ValueError(f"Scenario implementation changed: {rel}")
        if not isinstance(scenario["parameters"], dict):
            raise ValueError(f"{scenario['scenario_id']}: parameters must be explicit")
        if not scenario["lookback"] or not scenario["entity_key"]:
            raise ValueError(f"{scenario['scenario_id']}: missing lookback/entity key")
    return {"verified": True, "scenario_count": 11, "scenario_ids": ids}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--repository-root", type=Path, default=Path("."))
    args = p.parse_args()
    print(json.dumps(verify_manifest(json.loads(args.manifest.read_text()),
                                     args.repository_root), indent=2))


if __name__ == "__main__":
    main()
