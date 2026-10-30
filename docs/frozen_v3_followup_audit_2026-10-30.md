# Frozen V3 / XGBoost audit follow-up — 2026-10-30

**Evidence categories:** source inspected, GitHub Actions executed, reported frozen historical benchmark, and unverified provenance are explicitly separate.

## Newly recovered original 11-control V3

The historical branch `feature/payment-method-segmentation` at
`e5bc2c6a6aede7158594d01d6619a3febb8cb510` contains
`governance/PORTFOLIO_FREEZE_V3.md` (freeze 2026-10-02), the exact
11 control names, `src/materialize_frozen_v3_holdout.py`,
`src/materialize_deposit_send_holdout_spark.py`,
`src/materialize_governed_geo_holdout_spark.py`, and
`src/evaluate_frozen_v3_holdout.py`. This **supersedes** the earlier
statement that no original 11-scenario implementation had been located.
The source is on a historical branch, **not** yet a newly reproduced
HOLDOUT benchmark. The `main` `src/rule_based.py` contains 12 generic
flags; `feature/scenario-engine` has a separate four-pattern prototype.

## Critical: temporal semantics and potential lookahead

1. **Whole-bucket features in historical frozen V3 (high):**
   `src/materialize_frozen_scenarios.py:sender_windows` aggregates
   all rows in a fixed sender window and merges the resulting aggregate
   back to each row. `receiver_windows` does the same for receiver
   fixed buckets. An early transaction may thus be flagged using
   *later* transactions in its bucket. This can be valid as
   retrospective/batch detection **after window close**, but cannot be
   called an as-of-event real-time detector. The historical counts
   should not be changed without a separately versioned causal rerun.
2. **Forward 72-hour Deposit-Send (high):**
   `src/materialize_deposit_send_holdout_spark.py` explicitly joins
   subsequent outflows in `(event_time, event_time + 72h]`. This is a
   retrospective detection rule with a 72-hour maturity delay, not an
   immediately available feature. End-of-HOLDOUT rows are right-censored
   unless additional post-HOLDOUT events are provided.
3. **Geographic sanctions semantics (high, data dependent):**
   `src/build_governed_country_risk_spark.py` computes
   `is_sanctioned` as **nonempty** `Sanction` cells. The legacy
   `src/baseline_models.py` similarly uses `Sanction.notnull()`.
   If the original CSV uses textual negatives like `No` or `False`,
   those would be incorrectly treated as sanctioned. This is a
   *conditional defect*, not a measured misclassification: the original
   `data/country_risk.csv` is not tracked in the inspected branch.
   The new read-only `src/audit_country_risk_reference.py` detects
   this ambiguity without modifying the frozen scoring logic.
4. **Historical supervised geography target rates (high, unresolved):**
   The frozen XGBoost V1 feature manifest and original R target-rate
   construction are absent. `PastOnlyCountryEncoder` in the newer
   experimental V2 code uses earlier TRAIN timestamps and freezes
   mapping for DEVELOPMENT/TEST; it is **not** proof that V1 did so.
   Recover source code, fitted encoding, effective dates and feature
   order; demonstrate no HOLDOUT label access.
5. **TRAIN / DEVELOPMENT / HOLDOUT (medium):**
   `src/create_temporal_split.py` now validates dates and chooses
   date-separated 60/20/20-style boundaries. The exact frozen
   boundary manifest, source hash, row-level ID map and split outputs
   are not committed. The V2 trainer enforces strict time and ID
   disjointness, but its three model modes are **experimental**, not V1.
6. **XGBoost provenance (high):** the V2 experimental trainer declares
   `n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8,
   colsample_bytree=.8, random_state=42, tree_method=hist`. The
   legacy exploratory baseline uses `100/6/.1` and a random split.
   Neither set is authenticated as frozen XGBoost V1. Threshold
   `0.653366` is reported; DEVELOPMENT selection logs are missing.
   PR-AUC / average precision requires per-row scores and labels;
   no number can be inferred from the single confusion matrix.

## Historical numbers: reported and arithmetically consistent only

| HOLDOUT | Frozen 11-rule V3 | XGBoost V1 |
|---|---:|---:|
| Transactions | 1,898,009 | 1,898,009 |
| Positives | 2,136 | 2,136 |
| Flagged | 127,981 | 39,409 |
| True positives | 1,225 | 1,798 |
| False positives | 126,756 | 37,611 |
| False negatives | 911 | 338 |
| Precision | 0.9572% | 4.5624% |
| Recall | 57.3502% | 84.1760% |
| Alert rate | 6.7429% | 2.0763% |

These are **not newly rerun** from original HOLDOUT flags and model scores.
ML operational counts (39,409 transaction flags -> 35,107 alerts ->
30,114 cases) are at different grains; they require lineage reconciliation.

## Verified test execution and limits

GitHub Actions run
[40930084391](https://github.com/filippozonta2-commits/aml_transaction_monitoring_system/actions/runs/40930084391)
on commit `88d7a8afcf4ed01d6d585dbc93c66495a4b546cd`
completed successfully on 2026-10-29. The job logs show
**44 offline unittest cases passed**, Python source compilation and
published-count arithmetic reconciliation. These tests include V2
causal-feature, country-encoder, schema, split, adapter and frozen
manifest guards. They do **not** run original SAML-D, Spark, XGBoost
V1, the full Streamlit UI or signed SIT/UAT. The additional
country-risk reference tests were committed later; their CI outcome
must be checked separately.

## Exact next execution gates

- Recover the original governed `data/country_risk.csv` with a source
  date and semantics for the `Sanction` column. Run
  `python -m src.audit_country_risk_reference --country-risk ...`.
- Recover the original chronological split files and source hashes.
  Run `src/preflight_frozen_v3.py` and the **historical branch**
  materializers with Spark; preserve row-ID alignment and full logs.
- Reconcile 127,981 rule flags / 1,225 true positives without
  modifying the frozen thresholds. Separately label retrospective
  lookahead versus event-time causal controls.
- Recover XGBoost V1 model, exact feature list, hyperparameters,
  preprocessing, country target encoding, DEVELOPMENT threshold
  selection, and per-transaction HOLDOUT scores. Compute PR-AUC,
  PR curve and overlap **only** from that evidence.
- Execute formal SIT/UAT with fixtures, screenshots, expected vs
  actual, timestamps, defect IDs and sign-off; test dashboard artifact
  presence, orphan references, case aggregation and scenario switching.
- Batch roadmap: versioned source/feature/model/threshold manifests,
  data quality and schema gates, window-close scheduling, idempotent
  alerts/cases, replay and reconciliation. Streaming roadmap:
  event-time state, 72-hour maturation, watermarks, late events,
  offline-online parity, deduplication and latency tests. Neither
  deployment is evidenced as production.

**Freeze policy:** no original HOLDOUT metric, threshold or scenario
configuration is changed by this audit.
