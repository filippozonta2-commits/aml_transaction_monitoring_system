# SAML-D end-to-end simplified XGBoost benchmark — 2026-10-29

**Status: independently executed in the analysis runtime, NOT the full Hybrid V2.**
This is a new benchmark and must not replace the frozen V1 figures.

The user-provided SAML-D.csv contained **9,504,852** rows. We processed all rows into label-free transaction and simplified scenario features. Chronological day-boundary split:
- TRAIN: 5,707,316 rows; random 1,000,000-row subset used to fit each model.
- DEVELOPMENT: 1,899,527 rows; random 300,000-row subset used to set each threshold.
- HOLDOUT: **1,898,009 rows, 2,136 AML-positive**, all rows scored.

XGBoost: 120 trees, max depth 5, learning rate 0.07, subsample 0.85, column subsample 0.9, histogram tree method, seed 20261029. No hyperparameter tuning. Input features: log amount, cross-border, currency mismatch, payment type (label-independent integer category), and five simplified scenario flags/count/score or interactions. The rule-only model has few discrete scores, so its DEV-calibrated threshold does not guarantee a 2% alert rate on HOLDOUT.

## Measured HOLDOUT results at fixed 2% operational capacity

Exactly **37,960 alerts** selected per model, sorted by score with source-row-order tie-breaking:

| Model | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Transaction-only | 522 | 37,438 | 1,614 | 1.3751% | 24.4382% |
| Simplified rules + XGBoost | 153 | 37,807 | 1,983 | 0.4031% | 7.1629% |
| Simplified hybrid | 524 | 37,436 | 1,612 | 1.3804% | 24.5318% |

At the DEVELOPMENT-derived thresholds, the three HOLDOUT alert counts were 38,164; 174,663; 39,013 respectively. This difference illustrates alert-rate drift and extensive tied scores in the rule-only model. Operational top-k uses no HOLDOUT labels to select alerts.

**Interpretation:** The simplified hybrid only detects two more AML-positive transactions than the transaction-only baseline at equal investigator capacity. No meaningful improvement claim is supported by this experiment alone.

## Important limitations

- **Not** a run of the full `src/train_hybrid_v2.py` or `src/prepare_hybrid_v2.py` pipeline.
- No 24h/7d causal rolling or network features; an attempted full-history extraction was interrupted and did not yield usable results.
- No independent event-time country-risk or sanctions reference was available; those scenarios were omitted rather than guessed.
- No use of `Laundering_type` as a predictor or of labels in feature construction.
- Train and development were uniformly subsampled *within* their chronological windows; the entire HOLDOUT was scored.
- Payment categories were integer-coded by first occurrence, not target encoded.
- These results are not comparable feature-for-feature with the frozen V1, and no statistical confidence interval or repeated-seed robustness study was run.
- Source data and fitted model binaries are **not committed** to the repository.

A local reproduction package (Python scripts, fitted XGBoost models, result manifest) was produced separately. For the full Hybrid V2 benchmark, implement a memory-bounded, same-timestamp-isolated 24h/7d feature pipeline and verify external reference-data provenance before evaluation.
