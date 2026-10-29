"""Leakage-safe chronological target encoding for NEW experiments (not frozen V1).

TRAIN rows are encoded using only earlier timestamps. DEVELOPMENT and test
are transformed with mappings fitted exclusively on TRAIN. No test labels are
accepted by the transform method.
"""
from __future__ import annotations

import pandas as pd


class PastOnlyCountryEncoder:
    def __init__(self, smoothing: float = 20.0, prior: float = 0.0):
        if smoothing < 0 or not 0 <= prior <= 1:
            raise ValueError("Invalid smoothing or prior")
        self.smoothing = float(smoothing)
        self.prior = float(prior)
        self.mapping_: dict[str, float] | None = None
        self.train_prior_: float | None = None

    @staticmethod
    def _validate(frame: pd.DataFrame, country: str, time: str) -> pd.DataFrame:
        if frame[[country, time]].isna().any().any():
            raise ValueError("Country and event time cannot be missing")
        result = frame.copy()
        result[time] = pd.to_datetime(result[time], errors="raise", utc=True)
        if result[time].isna().any():
            raise ValueError("Missing event time")
        return result

    def fit_transform_train(self, frame: pd.DataFrame, *,
                            country: str = "country", label: str = "label",
                            time: str = "event_time") -> pd.Series:
        data = self._validate(frame, country, time)
        if data[label].isna().any() or not data[label].isin([0, 1]).all():
            raise ValueError("Expected binary nonmissing training labels")
        if not data[time].is_monotonic_increasing:
            raise ValueError("TRAIN must be sorted by event time")
        y = data[label].astype(float)
        # A shared timestamp is one simultaneous batch: labels from any row
        # at that timestamp cannot influence another row at the same time.
        prior = self.prior
        totals: dict[str, tuple[int, float]] = {}
        encoded = pd.Series(index=data.index, dtype=float, name="country_target_rate")
        for _, group in data.groupby(time, sort=True):
            for idx, row in group.iterrows():
                n, hits = totals.get(str(row[country]), (0, 0.0))
                encoded.loc[idx] = (hits + self.smoothing * prior) / (n + self.smoothing) if n + self.smoothing else prior
            for key, part in group.groupby(country, sort=False):
                k = str(key)
                n, hits = totals.get(k, (0, 0.0))
                totals[k] = (n + len(part), hits + float(part[label].sum()))
        self.train_prior_ = float(y.mean())
        self.mapping_ = {
            k: (hits + self.smoothing * self.train_prior_) / (n + self.smoothing)
            for k, (n, hits) in totals.items()
        }
        return encoded

    def transform(self, frame: pd.DataFrame, *, country: str = "country",
                  time: str = "event_time") -> pd.Series:
        if self.mapping_ is None or self.train_prior_ is None:
            raise RuntimeError("Fit on TRAIN before transforming")
        data = self._validate(frame, country, time)
        return data[country].astype(str).map(self.mapping_).fillna(
            self.train_prior_
        ).rename("country_target_rate")
