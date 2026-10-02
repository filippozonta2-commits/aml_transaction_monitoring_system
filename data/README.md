# Data

Raw datasets are not committed to this repository.

## Expected local files

Place the project datasets in this directory when running the pipeline locally:

- `SAML-D.csv` — transaction-level SAML-D data.
- `country_risk.csv` — country-level AML/financial-crime risk data used for enrichment.

The source code should reference these files through relative paths or configurable command-line arguments rather than machine-specific absolute paths.

> Do not commit large raw datasets, credentials, or sensitive financial data to this repository.
