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

## Repository branch recovery (verified)

An inspection of the GitHub branch `feature/scenario-engine` at commit
`3a196a7d1ee8e0c1e8400d913802de486565dbea` recovered the following
historical implementation files absent from `main`:

- `src/scenario_features_v3.py` — sender rolling feature prototype
- `src/receiver_features_v3.py` — receiver rolling feature prototype
- `src/scenario_engine_v3.py` — **explicitly four scenarios**: Smurfing, Fan-Out, Structuring, Fan-In
- `src/evaluate_holdout_v3.py` — **explicitly four-scenario** HOLDOUT evaluation
- `src/benchmark_scenarios_v3.py` — two-pattern DEVELOPMENT benchmark

**Important:** Despite the V3 filenames, these sources are NOT the original
11-scenario frozen benchmark. They must not be imported as an 11-scenario
reproduction. The recovered feature implementations can be independently
reviewed for causality and efficiency, but their provenance and thresholds
remain separate from the original frozen benchmark.

The repository's documented 11-scenario performance remains a historical
reference with insufficient executable provenance for exact re-scoring.

## Correct frozen portfolio located (supersedes earlier missing-source statement)

**FOUND:** `feature/payment-method-segmentation` at commit
`e5bc2c6a6aede7158594d01d6619a3febb8cb510` contains the actual
11-control DEVELOPMENT freeze dated 2026-10-02:
`governance/PORTFOLIO_FREEZE_V3.md`, `src/scenario_config_frozen.py`,
`src/scenario_catalog.py`, `src/materialize_frozen_v3_holdout.py`,
`src/materialize_deposit_send_holdout_spark.py`,
`src/materialize_governed_geo_holdout_spark.py`, and
`src/evaluate_frozen_v3_holdout.py`.

Original controls: STRUCTURING, DEPOSIT_SEND, FAN_OUT, FAN_IN,
CASH_WITHDRAWAL, SMURFING, UNUSUAL_AMOUNT_Z4,
SINGLE_LARGE_TRANSACTION, GATHER_SCATTER,
HIGH_RISK_GEOGRAPHY, SANCTIONED_GEOGRAPHY.

**Reproduction caution:** the frozen materializers use fixed windows with
whole-bucket aggregates (potential future-event visibility) and Deposit-Send
uses a forward 72-hour horizon (intentional delayed detection). These
semantics must be preserved for a historical reproduction, but audited
separately before calling the system real-time causal monitoring. Geography
requires the governed country-risk reference. No new HOLDOUT run has been
completed in this audit.

## Execution dependency audit (2026-10-29)

The original frozen V3 branch provides the 11-control materializers and final
HOLDOUT evaluator, but its tracked `data/` directory contains only README.md.
The required `data/country_risk.csv` is **not tracked** there. The country-risk
builder expects columns `Country`, `ISO Code`, `Overall Score`,
`ML/TF Risk`, `FATF`, and `Sanction`, with policy cutoff 6.50.
Do not fabricate this reference: doing so changes the frozen geography controls.

Historical execution order (once original inputs and dependencies are present):

```bash
python src/build_governed_country_risk_spark.py --country-risk data/country_risk.csv
python src/materialize_frozen_v3_holdout.py
python src/materialize_deposit_send_holdout_spark.py
python src/materialize_governed_geo_holdout_spark.py
python src/evaluate_frozen_v3_holdout.py
```

The historical branch uses some unqualified imports (e.g.
`from materialize_frozen_scenarios import ...`), so its scripts are typically
invoked as `python src/<script>.py` from the repository root, not necessarily
via `python src/<module>`. Adapt the invocation to the original imports;
do not change rule logic. The evaluation requires all three materialized
HOLDOUT parts to share identical `holdout_row_id` ordering.

**Audit limitation:** no full run has been executed in this session; the local
runtime does not have PySpark installed. Do not claim reproducibility from
source discovery alone.
