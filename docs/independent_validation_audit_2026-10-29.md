# Independent repository validation audit - 2026-10-29

**Scope:** `filippozonta2-commits/aml_transaction_monitoring_system`, default branch snapshot inspected on 2026-10-29. This audit documents what the *public repository* supports; it does not overwrite the previously reported frozen model or threshold. Educational/simulated SAML-D project.

## Executive decision

**Status: conditional / not yet independently reproducible.** The published HOLDOUT transaction counts are **arithmetically consistent**, but the public repository does not contain the frozen XGBoost training/scoring scripts, serialized model, exact feature schema, training configuration, development operating-point evidence, complete row-level HOLDOUT predictions, or source data needed to independently validate the reported **84.18% recall**. The claim should be described as a **reported frozen HOLDOUT result pending end-to-end independent replication**, not as a newly verified model run.

### Reported frozen transaction-level benchmark (not an independent rerun)

| Metric | Rule-Based V3 (11 scenarios) | XGBoost Detection V1 |
|---|---:|---:|
| HOLDOUT transactions | 1,898,009 | 1,898,009 |
| Labeled AML positives | 2,136 | 2,136 |
| Flagged transactions | 127,981 | 39,409 |
| TP | 1,225 | 1,798 |
| FP | 126,756 | 37,611 |
| FN | 911 | 338 |
| TN | 1,769,117 | 1,858,262 |
| Precision | 0.9572% | 4.5624% |
| Recall | 57.3502% | 84.1760% |
| Alert rate | 6.7429% | 2.0763% |

The ML-vs-rule difference is **+26.8258 percentage points of recall** and **69.21% fewer flagged transactions**. These calculations reconcile with `docs/mapping/04_holdout_evaluation.csv`; they **do not** prove the underlying labels/predictions or training protocol. ML operational counts (39,409 flagged transactions -> 35,107 aggregated alerts -> 30,114 cases) are a *different data grain* and were not independently rebuilt in this audit.

**PR-AUC / average precision:** not verifiable from confusion-matrix summaries or a single threshold. Requires one score per HOLDOUT transaction plus its label. **ROC-AUC:** likewise unavailable from the published counts. Neither should be inferred or fabricated.

## Critical findings and evidence boundaries

1. **Supervised geographic target-rate leakage: unresolved, high priority.** The personal website previously described country-level fraud/AML target rates built in R. No R script, fitted country-rate table, temporal out-of-fold encoding, or exact frozen model feature list is present in the inspected AML repository. If country target rates were calculated using DEVELOPMENT or HOLDOUT labels before training, that is direct target leakage. Even TRAIN-wide encodings for TRAIN rows can leak each row's own label unless generated out-of-fold or strictly past-only. **Action:** recover original R/Python transformation and provenance; recompute TRAIN encodings using chronological out-of-fold or past-only logic; fit mappings on TRAIN only for DEVELOPMENT/HOLDOUT; handle unknown countries with a TRAIN prior; rerun a newly versioned evaluation on an untouched later test period. Do not retroactively replace the frozen reported numbers.
2. **Country-risk enrichment is not the same as target encoding.** `src/baseline_models.py` merges `country_risk.csv` fields `Overall Score` and `Sanction` with country names; these columns are not explicitly computed from `Is_laundering` in the checked code. However, `data/country_risk.csv` is absent, so its origin, effective date, and potential target-derived construction cannot be independently audited. Distinguish exogenous geography-risk data from label-derived country fraud rates.
3. **Legacy baseline has non-temporal evaluation.** `src/baseline_models.py` calls `engineer_features(data)` on the full population **before** `train_test_split(..., stratify=y, random_state=42)`. This random split mixes historical and future observations; historical account features may include observations from rows allocated to the opposite partition. Therefore its exploratory metrics are **not** evidence for the frozen chronological XGBoost V1. `build_models` specifies exploratory XGBoost `n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=TRAIN imbalance, random_state=42`; these are **not verified frozen V1 hyperparameters**.
4. **Chronological split exists but execution evidence does not.** `src/create_temporal_split.py` defines TRAIN/DEVELOPMENT/HOLDOUT date boundaries using cumulative transaction counts. The split files, boundary manifest, date ranges, row-ID disjointness and model fit logs are not committed. In addition, rows with unparseable dates are silently excluded by the three masks; this needs fail-fast validation before any production-style rerun. Equal-timestamp ordering should use a documented deterministic tie key.
5. **Frozen model artifacts and exact feature schema missing.** The repo contains a 5-category/20-numeric feature list in `src/baseline_models.py`, but not an authenticated frozen XGBoost V1 feature manifest or training/scoring source. The paper's threshold `0.653366` is documented in dashboard and mapping, not backed here by a fitted model/DEVELOPMENT threshold-selection log. Threshold consistency can only be checked when row-level scores are supplied.
6. **Rule benchmark confusion in older documentation.** `docs/bsa_aml_regulatory_alignment.md` previously called a four-scenario result (66,012 flags / 525 hits) “frozen V3”; this is not the reported 11-scenario Rule-Based V3 (127,981 / 1,225). This audit branch corrects the distinction. `src/combined_v3.py` imports `src.scenario_features_v3`, which is not present in the inspected tree, so that legacy script cannot be assumed runnable.
7. **Dashboard is artifact-dependent.** The Streamlit console expects local CSVs under `results/` (including `results/holdout_v3/final_metrics.csv` and case-management tables). Those artifacts are not in the public tree; code-only checkout does not establish a working end-to-end demo. Some pages hardcode frozen population/threshold as display context. Model comparison reads saved row-level ML scores if available; it is not an independent model rerun.
8. **SIT/UAT evidence status.** Existing `tests/test_causal_features.py` contains three unittest cases for V2.2 causal sender features. The aggregation scripts implement defensive lineage checks; dashboard code checks orphaned IDs. These are implemented controls, **not a signed SIT/UAT execution cycle**. No controlled execution logs, screenshots, independent tester sign-off or formal UAT acceptance records were available. The newly added `tests/test_audit_frozen_metrics.py` has **three offline arithmetic/guardrail tests**; they were run successfully against a local copy of the new utility, but the existing suite and full pipeline were not executed against original source data during this audit.

## Reproduction checklist

- [ ] Recover source dataset version/hash, timestamp parsing convention, deterministic transaction row ID and exact TRAIN/DEVELOPMENT/HOLDOUT boundary manifest.
- [ ] Recover original country target-rate code, external country-risk source/provenance, all model feature names/types and preprocessing fit boundaries.
- [ ] Recover frozen XGBoost V1 training code, fitted model hash, hyperparameters, seed, dependency versions, class weighting, train/development metrics and threshold-selection record.
- [ ] Export `holdout_row_id, Is_laundering, RULE_ALERT_V3, ML_ALERT_V1, ml_probability` as a **joined, one-row-per-transaction** evidence file. Confirm flag-to-score consistency at threshold 0.653366, row uniqueness, and counts.
- [ ] Compute PR-AUC as `average_precision_score(y_true, ml_probability)`, full PR curve, confusion matrices, overlap, per-typology metrics, and uncertainty intervals on the frozen HOLDOUT; treat this as a diagnostic, not new threshold tuning.
- [ ] Rebuild rule and ML alert/case artifacts and reconcile evidence links, unique IDs, session boundaries and counts.
- [ ] Execute and preserve test logs, test data/fixtures, SIT/UAT expected-vs-actual results, screenshots and sign-off.

## Audit utility and execution

```bash
# Arithmetic reconciliation only (runs without raw transactions)
python -m src.audit_frozen_metrics

# Stronger audit, only when a real joined row-level evidence file is available
python -m src.audit_frozen_metrics --joined-row-level /path/to/joined_holdout_evidence.csv

# New offline unit tests (separate from original causal-feature tests)
python -m unittest tests.test_audit_frozen_metrics -v
```

**Important:** The row-level mode deliberately fails when the expected HOLDOUT population, frozen counts or ML threshold do not reconcile. It cannot reconstruct the missing model or establish the absence of geographic leakage; these require the original upstream artifacts.

## SIT / UAT and deployment roadmap

| Control | Status on 2026-10-29 | Next evidence |
|---|---|---|
| Published count arithmetic | Passed offline on supplied summary CSV | Row-level joined evidence |
| Causal V2.2 feature regression | Test source present, not rerun in this audit | Full suite log, pinned environment |
| Model training and threshold freeze | Claimed/documented, not independently verified | Model hash, config, DEV log |
| Alert/case foreign-key integrity | Defensive checks in code | Full artifacts, execution log |
| Investigator queue and network/geography | Streamlit source inspected | Dashboard smoke test and screenshots |
| SIT/UAT sign-off | Not evidenced | Formal test scripts, outcomes and approver |

**Batch first:** immutable transaction ingestion, schema/NaT checks, event-time feature computation, versioned feature/model/threshold manifests, idempotent alert/case writes, replay/reconciliation and monitored retries.

**Streaming later:** event-time watermarks, late/out-of-order events, stateful windows, deduplication, checkpoint/recovery, offline/online feature parity, latency/load testing and human case review. These are future designs, **not deployed capabilities**.

## Communication guidance

For recruiter/interview use: “The project reports 84.18% HOLDOUT recall at 4.56% precision versus 57.35% / 0.96% for frozen 11-scenario rules, on simulated transactions. Published counts reconcile arithmetically; independent reproducibility and feature-leakage review remain in progress.” Do not present the 84.18% as newly verified by this audit or cite a PR-AUC without the original scores.
