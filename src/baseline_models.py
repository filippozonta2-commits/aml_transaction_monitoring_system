"""Baseline machine-learning pipeline for AML transaction monitoring.

This module refactors the original exploratory prototype into reusable functions.
It trains Decision Tree, Random Forest, and XGBoost classifiers on SAML-D
transaction data enriched with country-level risk information.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from .features import add_transaction_features, add_historical_account_features
except ImportError:
    from features import add_transaction_features, add_historical_account_features


COUNTRY_MAP = {
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "USA": "United States",
}

CATEGORICAL_FEATURES = [
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_currency",
    "Payment_type",
    "Received_currency",
]

NUMERICAL_FEATURES = [
    "Log_amount",
    "Sender_risk_score",
    "Receiver_risk_score",
    "Cross_border",
    "Hour",
    "Day_of_week",
    "Is_weekend",
    "Is_night",
    "Currency_mismatch",
    "Risk_difference",
    "Max_country_risk",
    "Any_sanctioned_country",
    "Sender_prior_tx_count",
    "Receiver_prior_tx_count",
    "Sender_prior_total_amount",
    "Receiver_prior_total_amount",
    "Sender_prior_avg_amount",
    "Receiver_prior_avg_amount",
    "Sender_seconds_since_previous",
    "Receiver_seconds_since_previous",
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "Is_laundering"


def load_data(transaction_path: Path, country_risk_path: Path):
    """Load transaction and country-risk datasets."""
    transactions = pd.read_csv(transaction_path)
    country_risk = pd.read_csv(country_risk_path)
    return transactions, country_risk


def print_dataset_overview(data: pd.DataFrame) -> None:
    """Print lightweight exploratory diagnostics."""
    print("\nDataset overview:\n", data.head())
    print("\nColumn names:\n", data.columns.tolist())
    print("\nColumn datatypes:\n", data.dtypes)
    print("\nMissing values:\n", data.isnull().sum())
    print("\nDuplicated rows:\n", data.duplicated().sum())
    print("\nAML class distribution:\n", data[TARGET].value_counts())
    print(
        "\nAML class distribution (%):\n",
        data[TARGET].value_counts(normalize=True).mul(100),
    )


def enrich_country_risk(
    data: pd.DataFrame, country_risk: pd.DataFrame
) -> pd.DataFrame:
    """Attach sender/receiver country risk and sanction indicators."""
    data = data.copy()
    country_risk = country_risk.copy()

    data["Sender_bank_location"] = data["Sender_bank_location"].replace(COUNTRY_MAP)
    data["Receiver_bank_location"] = data["Receiver_bank_location"].replace(COUNTRY_MAP)

    country_risk["Is_sanctioned"] = np.where(
        country_risk["Sanction"].notnull(), 1, 0
    )
    country_risk = country_risk.rename(columns={"Overall Score": "Overall_score"})
    country_risk = country_risk[["Country", "Is_sanctioned", "Overall_score"]]

    sender_risk = country_risk.rename(
        columns={
            "Country": "Sender_bank_location",
            "Overall_score": "Sender_risk_score",
            "Is_sanctioned": "Sender_is_sanctioned",
        }
    )
    receiver_risk = country_risk.rename(
        columns={
            "Country": "Receiver_bank_location",
            "Overall_score": "Receiver_risk_score",
            "Is_sanctioned": "Receiver_is_sanctioned",
        }
    )

    data = data.merge(sender_risk, on="Sender_bank_location", how="left")
    data = data.merge(receiver_risk, on="Receiver_bank_location", how="left")
    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create transaction and leakage-safe historical account features."""
    data = add_transaction_features(data)
    data = add_historical_account_features(data)
    return data


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ]
    )


def build_models(y_train: pd.Series):
    """Create the three baseline classifiers."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="aucpr",
        ),
    }


def evaluate_model(name, pipeline, x_test, y_test):
    """Print classification metrics and return model probabilities."""
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, probabilities):.4f}")

    return probabilities


def threshold_analysis(y_test, probabilities, thresholds=(0.1, 0.2, 0.3, 0.4)):
    """Inspect precision/recall trade-offs at alternative alert thresholds."""
    print("\nThreshold analysis")
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = matrix.ravel()

        print(f"\nThreshold: {threshold:.2f}")
        print(matrix)
        print(f"TP={tp:,} | FP={fp:,} | FN={fn:,} | TN={tn:,}")
        print(classification_report(y_test, predictions, digits=4))


def save_xgboost_feature_importance(
    pipeline: Pipeline, output_path: Path, top_n: int = 20
) -> None:
    """Save XGBoost feature importance for the transformed feature space."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importance = classifier.feature_importances_

    feature_importance = (
        pd.DataFrame({"Feature": feature_names, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    print("\nTop feature importances:\n")
    print(feature_importance.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.barh(
        feature_importance["Feature"][::-1],
        feature_importance["Importance"][::-1],
    )
    plt.xlabel("Importance")
    plt.title(f"XGBoost - Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(transaction_path: Path, country_risk_path: Path) -> None:
    transactions, country_risk = load_data(transaction_path, country_risk_path)
    print_dataset_overview(transactions)

    data = enrich_country_risk(transactions, country_risk)
    data = engineer_features(data)

    print(
        "\nFraud rate: domestic vs cross-border\n",
        data.groupby("Cross_border")[TARGET].mean(),
    )

    x = data[FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = build_models(y_train)
    trained = {}

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        print(f"\nTraining {name}...")
        pipeline.fit(x_train, y_train)
        probabilities = evaluate_model(name, pipeline, x_test, y_test)
        threshold_analysis(y_test, probabilities)
        trained[name] = pipeline

    save_xgboost_feature_importance(
        trained["XGBoost"],
        Path("results/figures/xgboost_feature_importance.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baseline AML transaction-monitoring models."
    )
    parser.add_argument(
        "--transactions",
        type=Path,
        default=Path("data/SAML-D.csv"),
        help="Path to the SAML-D transaction CSV.",
    )
    parser.add_argument(
        "--country-risk",
        type=Path,
        default=Path("data/country_risk.csv"),
        help="Path to the country-risk CSV.",
    )
    args = parser.parse_args()

    main(args.transactions, args.country_risk)
