# AML Transaction Monitoring System

Python portfolio project on **simulated SAML-D transactions**, combining historical account behavior, rolling-window alert rules, and experimental supervised ML baselines. Large raw datasets are intentionally not committed.

## Results: four-scenario V3 (DEVELOPMENT only)

The causal rolling-window engine evaluates four transaction-monitoring patterns: **Smurfing, Fan-Out, Structuring and Fan-In**. On 1,899,527 DEVELOPMENT transactions (1,986 labeled AML), the current four-rule prototype produced:

| Metric | DEVELOPMENT result |
|---|---:|
| Alerts (deduplicated) | 66,215 |
| Alert rate | 3.49% |
| Labeled AML transactions detected | 503 / 1,986 |
| AML recall | 25.33% |
| Precision | 0.76% |

These are **development results, not holdout/generalization estimates**. Smurfing and Fan-Out thresholds were explored on DEVELOPMENT; receiver-side Structuring and Fan-In currently use initial V2-inspired thresholds. Precision remains low and the model is an experimental baseline, not a deployable monitoring system. V2 and V3 feature generation differ, so raw aggregate scores should not be construed as a controlled improvement.

## Reproduce locally

Install dependencies: `python3 -m pip install -r requirements.txt`.

Place temporal SAML-D splits at `data/temporal/SAML-D_train.csv` and `data/temporal/SAML-D_development.csv` (see `data/README.md`). Do **not** upload raw transaction data.

```bash
python3 -m unittest discover -s tests -v
python3 -m src.scenario_engine_v3
```

The program uses the preceding TRAIN transactions solely as warm-up history, evaluates on DEVELOPMENT, and saves `results/scenario_v3/development_summary.csv`. HOLDOUT is not read. Sender and receiver features use trailing time windows with stable ordering at equal timestamps; current transactions are included, future transactions are excluded.

## Code map

- `src/scenario_engine_v3.py` — combined four-rule V3 evaluation and deduplicated alerts
- `src/scenario_features_v3.py` — causal sender rolling features
- `src/receiver_features_v3.py` — causal receiver rolling features
- `src/tune_scenarios_v3.py` — exploratory DEVELOPMENT threshold sweep
- `src/combined_v3.py` — two-scenario trade-off exploration
- `tests/` — causal feature and window-boundary regression tests
- `src/scenario_engine_v2.py` — earlier fixed-bucket prototype, not a leakage-free benchmark equivalent

## U.S. BSA/AML regulatory interpretation

See [Regulatory alignment and limitations](docs/bsa_aml_regulatory_alignment.md) for scenario mapping to FinCEN/FFIEC guidance, the USA PATRIOT Act, and the distinction between electronic-transfer anomalies and cash structuring. **Rules are risk indicators, not statutory thresholds or automatic SAR decisions.**

## Frozen chronological HOLDOUT results

| Metric | HOLDOUT |
|---|---:|
| Transactions | 1,898,009 |
| Labeled AML transactions | 2,136 |
| Deduplicated alerts | 66,012 |
| AML hits | 525 |
| AML recall | 24.58% |
| Precision | 0.80% |
| Alert rate | 3.48% |

Reproduce with `python3 -m src.evaluate_holdout_v3`. These scores were measured before the regulatory documentation update; no rule thresholds were changed after observing HOLDOUT.

## Next milestones

- Freeze candidate rule thresholds and evaluate **once** on an untouched chronological HOLDOUT.
- Compare precision, recall, and recall at matched alert budgets against supervised ML models.
- Add operational dashboards/alert explanations only after evaluation quality is established.

## Disclaimer

Research/educational prototype using simulated data. No claim of operational AML effectiveness; not suitable for compliance decisions.
