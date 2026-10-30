"""Adapt original SAML-D columns to Hybrid V2 without target-derived risk.

Transaction IDs are stable source row ordinals, so duplicate business records
remain distinct. This module never reads Laundering_type for features.
"""
from __future__ import annotations

import pandas as pd

SOURCE_COLUMNS = (
    "Date", "Time", "Sender_account", "Receiver_account", "Amount",
    "Payment_currency", "Received_currency", "Sender_bank_location",
    "Receiver_bank_location", "Is_laundering",
)


def adapt_samld(frame: pd.DataFrame, *, start_row: int = 0) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing SAML-D columns: {missing}")
    if start_row < 0:
        raise ValueError("start_row must be nonnegative")
    result = pd.DataFrame(index=frame.index)
    result["transaction_id"] = range(start_row, start_row + len(frame))
    result["event_time"] = pd.to_datetime(
        frame["Date"].astype(str) + " " + frame["Time"].astype(str),
        utc=True, errors="raise",
    )
    if result["event_time"].isna().any():
        raise ValueError("Missing event timestamps")
    result["sender"] = frame["Sender_account"].astype("string")
    result["receiver"] = frame["Receiver_account"].astype("string")
    if result[["sender", "receiver"]].isna().any().any():
        raise ValueError("Missing sender or receiver")
    result["Amount"] = pd.to_numeric(frame["Amount"], errors="raise")
    if result["Amount"].isna().any() or (result["Amount"] < 0).any():
        raise ValueError("Invalid transaction amount")
    result["Is_laundering"] = pd.to_numeric(frame["Is_laundering"], errors="raise")
    if not result["Is_laundering"].isin([0, 1]).all():
        raise ValueError("Non-binary AML labels")
    sender_country = frame["Sender_bank_location"].astype("string")
    receiver_country = frame["Receiver_bank_location"].astype("string")
    payment_currency = frame["Payment_currency"].astype("string")
    received_currency = frame["Received_currency"].astype("string")
    if any(s.isna().any() for s in (
        sender_country, receiver_country, payment_currency, received_currency
    )):
        raise ValueError("Missing geographic or currency input")
    result["Cross_border"] = (sender_country != receiver_country).astype("int8")
    result["Currency_mismatch"] = (payment_currency != received_currency).astype("int8")
    # Do not guess sanctions or country-risk scores from observed AML labels.
    # These must be joined from independently versioned, as-of-event sources.
    return result.reset_index(drop=True)
