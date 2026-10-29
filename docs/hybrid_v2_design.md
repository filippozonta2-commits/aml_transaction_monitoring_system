# Hybrid AML Detection V2 — experimental design

**Status:** feature builder committed; not trained or benchmarked. Frozen V1 remains unchanged.

## Hypothesis

Rules can be useful as **predictive signals**, not necessarily final decisions. Construct 12 interpretable scenario indicators from as-of-transaction inputs and combine them with continuous transaction and behavioral features. Train a supervised detector to learn interactions and rank alerts. This does not eliminate the need for deterministic regulatory or sanctions controls; those may operate in a separate mandatory-control lane.

## Implementation

- `src/hybrid_features_v2.py`: `build_hybrid_features(frame)` reuses `src/rule_based.py` scenario logic and returns indicators, scenario intensity and available continuous history features. It excludes labels and `Scenario_alert` from the ML feature output.
- `tests/test_hybrid_features_v2.py`: label invariance, missing-input and partial network-input regression tests.
- `src/country_target_encoding_v2.py`: separate optional causal target encoding, never trained on test labels.

**Critical caveat:** `Sender_prior_*` and `*_window_*` are consumed, not created, by this builder. They must be generated with strictly historical event-time windows, with no current/future label or post-event investigation information. The current generic rule engine has 12 keys, **not** the historical frozen 11-rule benchmark; these are not interchangeable.

## Fair comparison

1. Frozen historical rule baseline (reported only until reproduced).
2. XGBoost with raw transaction features, trained anew on the same temporal TRAIN.
3. XGBoost with scenario indicators and continuous rule features only.
4. Hybrid XGBoost with raw plus rule-derived features.

Select hyperparameters and thresholds on DEVELOPMENT, freeze them, then evaluate once on a previously untouched temporal TEST. Compare average precision, recall at fixed alert rate, precision at fixed investigator capacity, typology-level performance and stability over time. Log feature lineage, split hashes, model hash, random seed and threshold. Report confidence intervals and do not imply production validation from synthetic-data experiments.

## Required next implementation

Build and test event-time causal rolling histories, training pipeline, development threshold selection, row-level test scoring and model registry manifests. The new builder alone does not establish causality of upstream inputs or any performance gain.
