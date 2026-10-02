"""Rolling-window and network features for AML scenario monitoring.

Features are computed from transaction history and are intended to support
configurable, focal-entity AML scenarios.
"""

import pandas as pd


def prepare_transaction_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    date_text = pd.to_datetime(result["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["Transaction_timestamp"] = pd.to_datetime(
        date_text + " " + result["Time"].astype(str), errors="coerce"
    )
    return result


def add_rolling_network_features(
    data: pd.DataFrame,
    lookback: str = "24h",
) -> pd.DataFrame:
    """Add historical rolling sender/receiver features.

    The rolling windows are closed on the left, so the current transaction is
    excluded from its own historical feature values.
    """
    result = prepare_transaction_timestamps(data)
    result["_original_order"] = range(len(result))
    result = result.sort_values("Transaction_timestamp", kind="stable")

    sender = result[
        ["Transaction_timestamp", "Sender_account", "Receiver_account", "Amount"]
    ].copy()
    sender = sender.sort_values(
        ["Sender_account", "Transaction_timestamp"], kind="stable"
    )

    sender["Sender_window_tx_count"] = (
        sender.set_index("Transaction_timestamp")
        .groupby("Sender_account")["Amount"]
        .rolling(lookback, closed="left")
        .count()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )
    sender["Sender_window_amount"] = (
        sender.set_index("Transaction_timestamp")
        .groupby("Sender_account")["Amount"]
        .rolling(lookback, closed="left")
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .to_numpy()
    )

    # Unique-counterparty counts require explicit historical windows.
    sender_unique = []
    for _, group in sender.groupby("Sender_account", sort=False):
        group = group.sort_values("Transaction_timestamp")
        timestamps = group["Transaction_timestamp"]
        receivers = group["Receiver_account"]
        values = []
        for i, ts in enumerate(timestamps):
            start = ts - pd.Timedelta(lookback)
            mask = (timestamps < ts) & (timestamps >= start)
            values.append(receivers[mask].nunique())
        sender_unique.extend(values)
    sender["Sender_window_unique_receivers"] = sender_unique

    receiver = result[
        ["Transaction_timestamp", "Receiver_account", "Sender_account", "Amount"]
    ].copy()
    receiver = receiver.sort_values(
        ["Receiver_account", "Transaction_timestamp"], kind="stable"
    )

    receiver["Receiver_window_tx_count"] = (
        receiver.set_index("Transaction_timestamp")
        .groupby("Receiver_account")["Amount"]
        .rolling(lookback, closed="left")
        .count()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )
    receiver["Receiver_window_amount"] = (
        receiver.set_index("Transaction_timestamp")
        .groupby("Receiver_account")["Amount"]
        .rolling(lookback, closed="left")
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .to_numpy()
    )

    receiver_unique = []
    for _, group in receiver.groupby("Receiver_account", sort=False):
        group = group.sort_values("Transaction_timestamp")
        timestamps = group["Transaction_timestamp"]
        senders = group["Sender_account"]
        values = []
        for i, ts in enumerate(timestamps):
            start = ts - pd.Timedelta(lookback)
            mask = (timestamps < ts) & (timestamps >= start)
            values.append(senders[mask].nunique())
        receiver_unique.extend(values)
    receiver["Receiver_window_unique_senders"] = receiver_unique

    sender_features = sender[
        [
            "Sender_window_tx_count",
            "Sender_window_amount",
            "Sender_window_unique_receivers",
        ]
    ]
    receiver_features = receiver[
        [
            "Receiver_window_tx_count",
            "Receiver_window_amount",
            "Receiver_window_unique_senders",
        ]
    ]

    result.loc[sender.index, sender_features.columns] = sender_features
    result.loc[receiver.index, receiver_features.columns] = receiver_features

    result = result.sort_values("_original_order").drop(columns="_original_order")
    return result
