"""Utilities for comparing rule-based and ML AML alerting."""

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_pred):
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Alerts": int(y_pred.sum()),
        "Alert_rate": float(y_pred.mean()),
    }


def compare_detection_engines(
    y_true,
    rule_alerts,
    ml_predictions,
    ml_probabilities,
) -> pd.DataFrame:
    """Return a compact comparison of rules and ML on identical observations."""
    rules = binary_metrics(y_true, rule_alerts)
    rules.update({"Model": "Rule-Based", "ROC_AUC": None, "PR_AUC": None})

    ml = binary_metrics(y_true, ml_predictions)
    ml.update(
        {
            "Model": "Machine Learning",
            "ROC_AUC": roc_auc_score(y_true, ml_probabilities),
            "PR_AUC": average_precision_score(y_true, ml_probabilities),
        }
    )

    columns = [
        "Model",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC",
        "Alerts",
        "Alert_rate",
    ]
    return pd.DataFrame([rules, ml])[columns]
