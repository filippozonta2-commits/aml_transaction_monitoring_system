import unittest

import pandas as pd

from src.country_target_encoding_v2 import PastOnlyCountryEncoder


class CountryTargetEncodingTests(unittest.TestCase):
    def test_no_current_or_future_labels(self):
        base = pd.DataFrame({
            "country": ["IT", "IT", "IT", "US"],
            "label": [1, 0, 1, 1],
            "event_time": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        })
        a = PastOnlyCountryEncoder(smoothing=2, prior=0.1).fit_transform_train(base)
        changed = base.copy()
        changed.loc[2:, "label"] = [0, 0]
        b = PastOnlyCountryEncoder(smoothing=2, prior=0.1).fit_transform_train(changed)
        self.assertEqual(a.iloc[0], b.iloc[0])
        self.assertEqual(a.iloc[1], b.iloc[1])
        self.assertEqual(a.iloc[2], b.iloc[2])

    def test_same_timestamp_labels_not_shared(self):
        frame = pd.DataFrame({
            "country": ["IT", "IT"], "label": [1, 0],
            "event_time": ["2023-01-01", "2023-01-01"],
        })
        out = PastOnlyCountryEncoder(prior=0.2).fit_transform_train(frame)
        self.assertEqual(out.iloc[0], 0.2)
        self.assertEqual(out.iloc[1], 0.2)

    def test_holdout_labels_never_read(self):
        train = pd.DataFrame({
            "country": ["IT", "US"], "label": [1, 0],
            "event_time": ["2023-01-01", "2023-01-02"],
        })
        encoder = PastOnlyCountryEncoder()
        encoder.fit_transform_train(train)
        holdout = pd.DataFrame({
            "country": ["IT", "XX"], "event_time": ["2024-01-01", "2024-01-02"],
            "label": [0, 1],
        })
        a = encoder.transform(holdout)
        holdout["label"] = [1, 0]
        b = encoder.transform(holdout)
        pd.testing.assert_series_equal(a, b)
        self.assertEqual(a.iloc[1], 0.5)

    def test_unsorted_train_rejected(self):
        frame = pd.DataFrame({
            "country": ["IT", "IT"], "label": [1, 0],
            "event_time": ["2023-01-02", "2023-01-01"],
        })
        with self.assertRaises(ValueError):
            PastOnlyCountryEncoder().fit_transform_train(frame)


if __name__ == "__main__":
    unittest.main()
