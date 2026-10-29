"""Causal, transaction-time rolling features for experimental AML Hybrid V2.

All historical windows exclude the current transaction and transactions with
the same timestamp. Rows must contain a unique transaction ID and valid times.
No labels are read. This reference implementation is intended for validation
and small datasets; large datasets require an equivalent streaming algorithm.
"""
from collections import defaultdict, deque
from datetime import timedelta
import pandas as pd

WINDOW_24H = timedelta(hours=24)
WINDOW_7D = timedelta(days=7)


def build_causal_rolling_features(
    frame: pd.DataFrame, *, id_col="transaction_id", time_col="event_time",
    sender_col="sender", receiver_col="receiver", amount_col="Amount",
) -> pd.DataFrame:
    required = [id_col, time_col, sender_col, receiver_col, amount_col]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("Missing IDs, timestamps, counterparties or amounts")
    if frame[id_col].duplicated().any():
        raise ValueError("Transaction IDs must be unique")
    data = frame.copy()
    data[time_col] = pd.to_datetime(data[time_col], utc=True, errors="raise")
    if data[time_col].isna().any():
        raise ValueError("Missing event time")
    data[amount_col] = pd.to_numeric(data[amount_col], errors="raise")
    if not data[amount_col].map(lambda x: pd.notna(x) and abs(x) != float("inf")).all():
        raise ValueError("Amounts must be finite")
    if (data[amount_col] < 0).any():
        raise ValueError("Expected nonnegative amounts")
    if not data[time_col].is_monotonic_increasing:
        raise ValueError("Input must be sorted by event time")

    sender_24h = defaultdict(deque)
    sender_7d = defaultdict(deque)
    receiver_7d = defaultdict(deque)
    sender_all = defaultdict(lambda: [0, 0.0, None])
    results = []

    def evict(q, cutoff):
        while q and q[0][0] < cutoff:
            q.popleft()

    # Process equal timestamps as simultaneous batches: do not allow same-time
    # records to influence each other's features.
    for timestamp, batch in data.groupby(time_col, sort=True):
        now = timestamp.to_pydatetime()
        for _, row in batch.iterrows():
            s, r = row[sender_col], row[receiver_col]
            a = sender_all[s]
            q24, q7, qr = sender_24h[s], sender_7d[s], receiver_7d[r]
            evict(q24, now - WINDOW_24H)
            evict(q7, now - WINDOW_7D)
            evict(qr, now - WINDOW_7D)
            results.append({
                id_col: row[id_col],
                "Sender_seconds_since_previous": (now - a[2]).total_seconds() if a[2] is not None else float("nan"),
                "Sender_prior_tx_count": a[0],
                "Sender_prior_avg_amount": a[1] / a[0] if a[0] else float("nan"),
                "Sender_24h_tx_count": len(q24),
                "Sender_24h_amount": sum(item[1] for item in q24),
                "Sender_window_tx_count": len(q7),
                "Sender_window_amount": sum(item[1] for item in q7),
                "Sender_window_unique_receivers": len({item[2] for item in q7}),
                "Receiver_window_tx_count": len(qr),
                "Receiver_window_amount": sum(item[1] for item in qr),
                "Receiver_window_unique_senders": len({item[2] for item in qr}),
            })
        for _, row in batch.iterrows():
            s, r, amount = row[sender_col], row[receiver_col], float(row[amount_col])
            sender_24h[s].append((now, amount, r))
            sender_7d[s].append((now, amount, r))
            receiver_7d[r].append((now, amount, s))
            sender_all[s][0] += 1
            sender_all[s][1] += amount
            sender_all[s][2] = now
    return pd.DataFrame(results, columns=[id_col,
        "Sender_seconds_since_previous", "Sender_prior_tx_count",
        "Sender_prior_avg_amount", "Sender_24h_tx_count", "Sender_24h_amount",
        "Sender_window_tx_count", "Sender_window_amount",
        "Sender_window_unique_receivers", "Receiver_window_tx_count",
        "Receiver_window_amount", "Receiver_window_unique_senders",
    ])
