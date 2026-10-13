# AML reproducible pipeline

Run from repository root:

```bash
pip install -r requirements.txt
python -m src.aml_pipeline --input data/raw/SAML-D.csv --output results/pipeline
```

The pipeline loads a SAML-D-style transaction CSV, sorts chronologically, builds transaction and geographic features, trains a class-weighted logistic regression baseline, selects an F1 threshold on development data, and evaluates the untouched temporal holdout. Results include `metrics.json` and `holdout_alerts.csv`.

**Limitations:** This is a baseline, not a finished AML system. It does not yet implement the V2.2 rule engine, country-risk lookup, network features, XGBoost comparison, or Streamlit dashboard. No results are claimed until run against the local dataset. The holdout must not be reused for iterative tuning.
