"""Causal receiver-side rolling features for AML scenarios."""
from collections import Counter, deque
import pandas as pd


def receiver_rolling_features(df, window_days):
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    out = df.sort_values("Transaction_timestamp", kind="stable").copy()
    if out["Transaction_timestamp"].isna().any():
        raise ValueError("Missing transaction timestamps")
    window = pd.Timedelta(days=window_days)
    names = ("receiver_tx_count", "receiver_aggregate_amount",
             "receiver_unique_senders", "receiver_cross_border_rate",
             "receiver_currency_mismatch_rate")
    values = {name: [] for name in names}
    states = {}
    for row in out.itertuples(index=False):
        receiver, sender, stamp = row.Receiver_account, row.Sender_account, row.Transaction_timestamp
        amount = float(row.Amount)
        border, fx = int(row.Cross_border), int(row.Currency_mismatch)
        if receiver not in states:
            states[receiver] = {"events": deque(), "senders": Counter(),
                                "amount": 0., "border": 0, "fx": 0}
        s = states[receiver]
        while s["events"] and s["events"][0][0] < stamp - window:
            old_stamp, old_sender, old_amount, old_border, old_fx = s["events"].popleft()
            s["amount"] -= old_amount
            s["border"] -= old_border
            s["fx"] -= old_fx
            s["senders"][old_sender] -= 1
            if not s["senders"][old_sender]:
                del s["senders"][old_sender]
        s["events"].append((stamp, sender, amount, border, fx))
        s["senders"][sender] += 1
        s["amount"] += amount
        s["border"] += border
        s["fx"] += fx
        n = len(s["events"])
        for name, value in zip(names, (n, s["amount"], len(s["senders"]),
                                      s["border"] / n, s["fx"] / n)):
            values[name].append(value)
    for name, v in values.items():
        out[name] = v
    return out
