# Frozen Rule-Based V3: provenance and reproduction gate

**The 11 original frozen scenarios have NOT yet been independently reconstructed.**
The historical summary is: HOLDOUT 1,898,009 rows; AML positives 2,136;
flagged 127,981; true positives 1,225; false positives 126,756;
false negatives 911; recall 57.35%; precision 0.96%.

The integrated white paper documents the 11-scenario count and summary,
but does not supply the exact eleven executable rules, thresholds, lookback
windows, and source hashes. `src/rule_based.py` currently defines **12**
scenario flags, while `src/scenario_engine_v2.py` implements **four**
experimental V2 patterns. Neither may be relabeled as the frozen V3.

`python -m src.verify_frozen_v3 --manifest PATH` verifies a user-supplied
11-scenario source manifest. It requires exactly 11 unique IDs, a
repository-relative implementation path, SHA-256 checksum, explicit parameter
dictionary, lookback and entity key for every scenario. **Passing this gate
verifies source integrity and completeness, not numerical reproduction.**

To claim reproduction, also execute all 11 original scenarios against the
same event-time HOLDOUT and independently reconcile transaction-level flags
to the historical counts. No unverified scenario should be substituted,
invented or dropped. Historical summary metrics remain reference-only until
the row-level evidence and original scenario implementation are available.

The simplified benchmark in `docs/samld_simplified_benchmark_2026-10-29.md`
was rejected by the project owner and is not a valid V3 or Hybrid V2 result.
