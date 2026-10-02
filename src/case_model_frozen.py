"""Frozen case-prioritization model specification.

This file is the immutable DEVELOPMENT-selected contract used for the final
single-shot HOLDOUT evaluation. Do not tune it from HOLDOUT results.
"""
CASE_POLICY={"type":"inactivity_gap","hours":72}
TEMPORAL_TRAIN_FRACTION=0.70
DROP_REDUNDANT={"amount_per_alert","currency_mismatch_rate","multi_scenario_flag",
                "payment_share_Cross-border","scenario_events","total_alert_amount"}
MODEL_NAME="xgboost"
MODEL_PARAMS={
    "n_estimators":350,
    "max_depth":4,
    "learning_rate":0.05,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "min_child_weight":3,
    "reg_lambda":1.0,
    "objective":"binary:logistic",
    "eval_metric":"logloss",
    "random_state":42,
    "n_jobs":-1,
}
PRIMARY_METRIC="average_precision"
RANKING_K=(100,250,500,1000)
VALIDATION_REFERENCE={
    "account_purged_cases":2850,
    "account_purged_positives":160,
    "roc_auc":0.904778,
    "pr_auc":0.440964,
    "precision_at_250":0.348,
    "recall_at_250":0.54375,
}
FREEZE_NOTE=(
    "Selected entirely from TRAIN + DEVELOPMENT. HOLDOUT must not be used to "
    "change case policy, feature definitions, feature exclusions, model "
    "hyperparameters, thresholds, or reported primary metrics."
)
