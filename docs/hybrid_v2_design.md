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

## Causal rolling feature reference implementation

`src/causal_rolling_features_v2.py` now computes prior sender counts/averages, 24-hour velocity and amount, and 7-day sender/receiver counts, amounts and distinct counterparties. It processes equal-timestamp events as simultaneous batches and excludes future events. Five focused regression tests cover same-time isolation, window expiry, label invariance, future-row invariance and duplicate IDs. **Limitations:** this is an in-memory reference implementation with per-group Python loops, not yet optimized for the full 1.9M+ transaction dataset; it must be benchmarked and compared against an independently implemented oracle before production claims. It requires globally sorted input and a documented choice of event-time semantics.

## Split preparation (new)

`src/prepare_hybrid_v2.py` creates TRAIN/DEVELOPMENT/TEST CSVs from **one already sorted event-time source**. It computes causal rolling histories before partitioning, preserving valid prior context across split boundaries; three tests check continuity and invalid input. Example:

```bash
python -m src.prepare_hybrid_v2 --input data/v2/source.csv --train-end 2023-05-01 --development-end 2023-07-01 --output-dir data/v2
```

Input must contain transaction_id, event_time, sender, receiver, Amount, Is_laundering, Max_country_risk, Any_sanctioned_country, Cross_border and Currency_mismatch. The repository currently does **not** include the raw SAML-D CSV; this reference preparer is not yet a validated SAML-D column adapter. Source risk and geographic inputs also require independent as-of-event lineage checks. The script is in-memory and must be optimized before full-scale use.

## Experimental trainer (new)

`src/train_hybrid_v2.py` trains three XGBoost ablations on separately supplied, already causally engineered TRAIN/DEVELOPMENT/TEST CSVs. It enforces chronological nonoverlap, unique IDs, two-class splits, identical feature schemas, and chooses each operating threshold on DEVELOPMENT under a configurable alert budget (default 2%). It evaluates TEST after threshold selection and writes input SHA-256 hashes and metrics. `tests/test_train_hybrid_v2.py` checks split leakage and alert-budget behavior.

Example (requires real prepared data; no synthetic performance claims):

```bash
python -m src.train_hybrid_v2 --train data/v2/train.csv --development data/v2/development.csv --test data/v2/test.csv --max-alert-rate 0.02
```

**Important:** the script assumes its input features are genuinely as-of-event. It does not yet build or audit those histories end-to-end, and running repeated model iterations against the same TEST would invalidate its untouched-test assumption. For a defensible final result, reserve TEST for a single frozen evaluation and persist model files and preprocessing manifests.

## Required next implementation

Build and test event-time causal rolling histories, training pipeline, development threshold selection, row-level test scoring and model registry manifests. The new builder alone does not establish causality of upstream inputs or any performance gain.
