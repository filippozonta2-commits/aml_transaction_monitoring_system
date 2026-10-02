"""Standard alert records for the AML scenario engine."""

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd


@dataclass
class AlertRecord:
    alert_id: str
    scenario_id: str
    scenario_name: str
    focal_account: str
    alert_timestamp: pd.Timestamp
    lookback_start: Optional[pd.Timestamp]
    lookback_end: Optional[pd.Timestamp]
    transaction_count: int
    aggregate_amount: float
    unique_counterparties: int
    scenario_score: int
    reason: str


def build_transaction_alerts(
    data: pd.DataFrame,
    scenario_id: str,
    scenario_name: str,
    flag_column: str,
    score: int,
    focal_column: str = "Sender_account",
) -> pd.DataFrame:
    """Convert transaction-level scenario hits into standardized alert records."""
    hits = data.loc[data[flag_column] == 1].copy()
    if hits.empty:
        return pd.DataFrame(columns=AlertRecord.__annotations__.keys())

    timestamp = hits.get("Transaction_timestamp")
    if timestamp is None:
        date_text = pd.to_datetime(hits["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        timestamp = pd.to_datetime(
            date_text + " " + hits["Time"].astype(str), errors="coerce"
        )

    records = []
    for sequence, (idx, row) in enumerate(hits.iterrows(), start=1):
        ts = timestamp.loc[idx]
        record = AlertRecord(
            alert_id=f"{scenario_id}-{sequence:07d}",
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            focal_account=str(row[focal_column]),
            alert_timestamp=ts,
            lookback_start=None,
            lookback_end=ts,
            transaction_count=1,
            aggregate_amount=float(row["Amount"]),
            unique_counterparties=1,
            scenario_score=score,
            reason=str(row.get("Alert_reasons", scenario_name)),
        )
        records.append(asdict(record))

    return pd.DataFrame(records)


def build_window_alerts(
    data: pd.DataFrame,
    scenario_id: str,
    scenario_name: str,
    flag_column: str,
    score: int,
    lookback: str,
    focal_column: str,
    tx_count_column: str,
    amount_column: str,
    counterparty_column: str,
) -> pd.DataFrame:
    """Convert rolling-window scenario hits into standardized alert records."""
    hits = data.loc[data[flag_column] == 1].copy()
    if hits.empty:
        return pd.DataFrame(columns=AlertRecord.__annotations__.keys())

    records = []
    delta = pd.Timedelta(lookback)

    for sequence, (_, row) in enumerate(hits.iterrows(), start=1):
        ts = row["Transaction_timestamp"]
        record = AlertRecord(
            alert_id=f"{scenario_id}-{sequence:07d}",
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            focal_account=str(row[focal_column]),
            alert_timestamp=ts,
            lookback_start=ts - delta,
            lookback_end=ts,
            transaction_count=int(row[tx_count_column]),
            aggregate_amount=float(row[amount_column]),
            unique_counterparties=int(row[counterparty_column]),
            scenario_score=score,
            reason=scenario_name,
        )
        records.append(asdict(record))

    return pd.DataFrame(records)
