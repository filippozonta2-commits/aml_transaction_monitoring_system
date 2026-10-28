"""Causal rolling-window scenario features (V3 experimental).

Computes features on TRAIN warm-up plus DEVELOPMENT; only DEVELOPMENT
transactions should be evaluated. No HOLDOUT access.
"""
from collections import Counter, deque
import heapq
import pandas as pd


def sender_rolling_features(df, window_days):
    """Trailing time windows, including the current transaction.

    Uses chronological order and original input order for equal timestamps.
    Keeps separate state for each sender; no future transactions are used.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if df["Transaction_timestamp"].isna().any():
        raise ValueError("Transaction_timestamp contains missing values")

    out = df.sort_values("Transaction_timestamp", kind="stable").copy()
    window = pd.Timedelta(days=window_days)
    count = len(out)
    features = {
        "tx_count": [0] * count,
        "aggregate_amount": [0.0] * count,
        "median_amount": [0.0] * count,
        "unique_receivers": [0] * count,
        "cross_border_rate": [0.0] * count,
        "currency_mismatch_rate": [0.0] * count,
    }
    states = {}
    for i, row in enumerate(out.itertuples(index=False)):
        sender = row.Sender_account
        stamp = row.Transaction_timestamp
        amount = float(row.Amount)
        receiver = row.Receiver_account
        border = int(row.Cross_border)
        mismatch = int(row.Currency_mismatch)
        if sender not in states:
            states[sender] = {
                "events": deque(), "receivers": Counter(),
                "lo": [], "hi": [], "active": Counter(),
                "amount": 0., "border": 0, "mismatch": 0,
                "next_id": 0,
            }
        s = states[sender]
        def prune(heap):
            while heap and s["active"][heap[0][1]] == 0:
                heapq.heappop(heap)

        def rebalance_median():
            # Heap median maintenance with deletions is error-prone.
            # V3 uses an exact, sorted active-window amount list below.
            return None

        # Exact time boundary: retain events with timestamp >= now - window.
        while s["events"] and s["events"][0][0] < stamp - window:
            old = s["events"].popleft()
            s["amount"] -= old[1]
            s["border"] -= old[3]
            s["mismatch"] -= old[4]
            s["receivers"][old[2]] -= 1
            if not s["receivers"][old[2]]:
                del s["receivers"][old[2]]
        s["events"].append((stamp, amount, receiver, border, mismatch))
        s["amount"] += amount
        s["border"] += border
        s["mismatch"] += mismatch
        s["receivers"][receiver] += 1
        n = len(s["events"])
        vals = sorted(event[1] for event in s["events"])
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        features["tx_count"][i] = n
        features["aggregate_amount"][i] = s["amount"]
        features["median_amount"][i] = median
        features["unique_receivers"][i] = len(s["receivers"])
        features["cross_border_rate"][i] = s["border"] / n
        features["currency_mismatch_rate"][i] = s["mismatch"] / n
    for name, values in features.items():
        out[name] = values
    return out
