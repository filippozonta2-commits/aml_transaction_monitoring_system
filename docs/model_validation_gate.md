# Model validation and reproducibility gate

**Status (2026-10-29): NOT independently reproducible.** This is an educational, simulated-transaction research project. The reported 84.18% recall / 4.56% precision is historical and must not be advertised as independently validated.

## Evidence required before promoting reported metrics to validated metrics

| Gate | Required evidence | Current status |
|---|---|---|
| Data lineage | Source version/hash, stable transaction IDs, event timestamps, deterministic TRAIN/DEVELOPMENT/HOLDOUT split manifest | Missing from public checkout |
| Feature provenance | Exact frozen feature schema; country-risk source and effective date; original R target-rate code and fitted mappings | Missing |
| Leakage review | Demonstrate that supervised country encodings do not use row labels, future labels or DEVELOPMENT/HOLDOUT labels; TRAIN encodings must be out-of-fold or strictly past-only | Unresolved, high priority |
| Frozen XGBoost V1 | Training and scoring source, dependency lock, seed, hyperparameters, serialized model hash | Missing |
| Threshold selection | DEVELOPMENT score/label file, operating-point selection criteria, threshold freeze record | Missing |
| Row-level HOLDOUT | One row per transaction with stable ID, label, rule flag, ML score and ML flag | Missing |
| Metrics | Reconcile confusion matrices, average precision / PR curve, overlap and typology breakdown from joined rows | Summary arithmetic only |
| Rule baseline | Exact frozen 11-scenario configuration and provenance; reconcile generic 12-key module | Unresolved |
| Operational lineage | Rebuild 7-day alert and 30-day case sessions, validate IDs and foreign keys | Not independently rerun |
| SIT/UAT | Versioned fixtures, execution evidence, expected vs actual results, reviewer sign-off | Not completed |

## Versioning protocol

- **V1 is immutable historical evidence.** Do not silently retrain it, substitute a baseline script, alter its threshold or overwrite its reported metrics.
- **V2 is a new experiment.** Recover and review feature construction first. Fit all preprocessing on TRAIN only; generate supervised TRAIN encodings chronologically or out-of-fold, then apply TRAIN-fitted mappings to DEVELOPMENT and test. Unseen countries use a documented TRAIN-derived prior.
- **Model selection belongs to DEVELOPMENT.** Freeze features, hyperparameters and threshold before final evaluation.
- **Prefer a new untouched later test period for V2** if the historical HOLDOUT has already influenced development. Record its dates and row IDs.
- **Publish uncertainty and limitations.** Report prevalence, precision, recall, alert rate, average precision, recall at fixed alert budgets, confidence intervals, and leakage checks. Never infer PR-AUC from a single confusion matrix.
- **No synthetic stand-ins for audit evidence.** Synthetic unit fixtures can test code, but cannot validate historical model performance.

## Suggested evidence layout (do not commit sensitive or oversized raw data)

```text
reproducibility/
  data_manifest.json
  split_manifest.json
  feature_manifest.json
  country_encoding_provenance.md
  model_v1_manifest.json
  development_threshold_selection.csv
  holdout_evaluation_manifest.json
  test_execution_log.md
```

Manifests should contain file hashes, versions, date boundaries, column names, counts and command lines; omit sensitive transaction content. Where files cannot be redistributed, explain access restrictions and provide reproducible generation steps.

## Implemented V2 building block (not a trained detector)

`src/country_target_encoding_v2.py` implements a past-only supervised country encoder. It treats equal event timestamps as simultaneous batches, never reads DEVELOPMENT/HOLDOUT labels in `transform`, and maps unseen countries to the TRAIN label prior. `tests/test_country_target_encoding_v2.py` contains four regression tests. The encoder is **not wired into a training pipeline**, and no V2 metrics have been produced. Run `python -m unittest tests.test_country_target_encoding_v2 -v` to execute the tests; CI results must be checked separately.

## Acceptance criteria for a validated V2

1. Unit tests demonstrate country encodings cannot see future or test labels, including previously unseen countries.
2. Split tests demonstrate deterministic boundaries, disjoint stable row IDs and no dropped/unparseable timestamps.
3. Training, threshold selection and scoring are reproducible from versioned source and pinned dependencies.
4. A single joined row-level test artifact reproduces every published detector count and calculates PR-AUC without tuning on test.
5. The exact frozen rule configuration is versioned and rule/ML metrics are computed on identical rows.
6. Alert/case referential integrity and session boundaries are tested; SIT/UAT logs are retained.
7. README, dashboard, white paper and personal website use identical validated numbers and clearly distinguish transaction, alert and case counts.

See [independent validation audit](independent_validation_audit_2026-10-29.md) for the evidence currently available.
