# ML DETECTION FREEZE V1

## Status
FROZEN before ML HOLDOUT evaluation.

## Purpose
Independent transaction-level machine-learning AML detection engine for comparison with Frozen Rule-Based V3.

## Development selection
- Model: XGBoost
- Training data: full TRAIN (5,707,316 transactions; 5,751 AML positives)
- Model selection and threshold selection: DEVELOPMENT only
- HOLDOUT used for selection/tuning: NO
- DEVELOPMENT ROC-AUC: 0.990728
- DEVELOPMENT PR-AUC: 0.536416

## Frozen operating point
- Probability threshold: **0.653366**
- DEVELOPMENT alert rate: ~2.00%
- DEVELOPMENT alerts: 37,991
- DEVELOPMENT AML hits: 1,733
- DEVELOPMENT precision: 4.5616%
- DEVELOPMENT recall: 87.2608%

The 2% operating point was selected as the efficiency operating point before ML HOLDOUT access. The 6.94% workload-matched point remains a diagnostic comparison only and is not the production threshold.

## Frozen model specification
- n_estimators: 120
- max_depth: 5
- learning_rate: 0.10
- subsample: 0.8
- colsample_bytree: 0.8
- tree_method: hist
- class weighting: scale_pos_weight derived from TRAIN
- random_state: 42
- eval_metric: aucpr

## Feature and preprocessing contract
Feature definitions are inherited from src/baseline_models.py and src/features.py.
Historical account features are prior-only. HOLDOUT labels must never be used to construct features or scores.
Categorical features use most-frequent imputation + one-hot encoding with unknown-category handling.
Numerical features use median imputation.

## Governance
This freeze prohibits changing model family, hyperparameters, features, preprocessing, or threshold in response to HOLDOUT results.
Any future redesign requires a new version and a new untouched validation protocol.
