"""Feature engineering utilities for AML transaction monitoring."""

import numpy as np
import pandas as pd


def add_transaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe transaction-level features available at alert time."""
    data = data.copy()

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    parsed_time = pd.to_datetime(data["Time"], format="%H:%M:%S", errors="coerce")

    data["Hour"] = parsed_time.dt.hour
    data["Day_of_week"] = data["Date"].dt.dayofweek
    data["Is_weekend"] = data["Day_of_week"].ge(5).astype(int)
    data["Is_night"] = data["Hour"].between(0, 5).astype(int)

    data["Cross_border"] = (
        data["Sender_bank_location"] != data["Receiver_bank_location"]
    ).astype(int)

    data["Currency_mismatch"] = (
        data["Payment_currency"] != data["Received_currency"]
    ).astype(int)

    data["Log_amount"] = np.log1p(data["Amount"])

    data["Risk_difference"] = (
        data["Sender_risk_score"] - data["Receiver_risk_score"]
    ).abs()

    data["Max_country_risk"] = data[
        ["Sender_risk_score", "Receiver_risk_score"]
    ].max(axis=1)

    data["Any_sanctioned_country"] = (
        data[["Sender_is_sanctioned", "Receiver_is_sanctioned"]]
        .fillna(0)
        .max(axis=1)
        .astype(int)
    )

    return data


def add_historical_account_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add account behavior using only transactions observed earlier in time.

    Rows are ordered by transaction timestamp. Shifted cumulative statistics
    prevent the current transaction (and later transactions) from contributing
    to its own behavioral features.
    """
    data = data.copy()

    date_text = pd.to_datetime(data["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    time_text = data["Time"].astype(str)
    data["Transaction_timestamp"] = pd.to_datetime(
        date_text + " " + time_text, errors="coerce"
    )

    original_index = data.index
    data = data.sort_values("Transaction_timestamp", kind="stable")

    sender_group = data.groupby("Sender_account", sort=False)
    receiver_group = data.groupby("Receiver_account", sort=False)

    data["Sender_prior_tx_count"] = sender_group.cumcount()
    data["Receiver_prior_tx_count"] = receiver_group.cumcount()

    sender_cumulative_amount = sender_group["Amount"].cumsum()
    receiver_cumulative_amount = receiver_group["Amount"].cumsum()

    data["Sender_prior_total_amount"] = sender_cumulative_amount - data["Amount"]
    data["Receiver_prior_total_amount"] = receiver_cumulative_amount - data["Amount"]

    data["Sender_prior_avg_amount"] = np.where(
        data["Sender_prior_tx_count"] > 0,
        data["Sender_prior_total_amount"] / data["Sender_prior_tx_count"],
        0.0,
    )
    data["Receiver_prior_avg_amount"] = np.where(
        data["Receiver_prior_tx_count"] > 0,
        data["Receiver_prior_total_amount"] / data["Receiver_prior_tx_count"],
        0.0,
    )

    data["Sender_seconds_since_previous"] = (
        sender_group["Transaction_timestamp"].diff().dt.total_seconds()
    )
    data["Receiver_seconds_since_previous"] = (
        receiver_group["Transaction_timestamp"].diff().dt.total_seconds()
    )

    data["Sender_seconds_since_previous"] = (
        data["Sender_seconds_since_previous"].fillna(-1)
    )
    data["Receiver_seconds_since_previous"] = (
        data["Receiver_seconds_since_previous"].fillna(-1)
    )

    return data.reindex(original_index)
