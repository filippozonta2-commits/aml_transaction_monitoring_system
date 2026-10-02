# AML Transaction Monitoring System

End-to-end anti-money laundering (AML) transaction monitoring project comparing traditional rule-based detection with machine-learning approaches, enriched with geographic and country-risk information and designed to culminate in an interactive investigation dashboard.

## Project objective

Traditional AML transaction-monitoring systems often generate large volumes of alerts through fixed rules. This project investigates whether machine-learning models can improve the detection of simulated money-laundering activity while reducing unnecessary alerts.

The project is being developed around the **SAML-D** transaction dataset and external country-level financial-crime risk information.

## Planned architecture

```text
Transaction data
      |
      +--> Data quality & exploratory analysis
      |
      +--> Country-risk enrichment
      |
      +--> Feature engineering
      |       - transaction features
      |       - geographic/currency features
      |       - temporal features
      |       - behavioral/network features
      |
      +-----------------------+
      |                       |
Rule-based engine      Machine-learning models
      |                       |
      +-----------+-----------+
                  |
           Model comparison
                  |
          Alert prioritization
                  |
       Investigation dashboard
```

## Current baseline

The current prototype includes:

- exploratory checks for missing values, duplicates, class balance, currencies and countries;
- cross-border transaction identification;
- sender and receiver country-risk enrichment;
- log-transformed transaction amount;
- categorical preprocessing with one-hot encoding;
- Decision Tree baseline;
- Random Forest classifier;
- XGBoost classifier with class-imbalance weighting;
- threshold analysis;
- feature-importance analysis.

## Evaluation strategy

Because AML detection is highly imbalanced, the project will emphasize metrics that reflect alert quality rather than accuracy alone:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC / Average Precision
- False-positive rate
- Recall at a fixed alert budget

The rule-based engine and ML models will ultimately be evaluated on the same holdout data.

## Roadmap

- [x] Initial transaction-level ML prototype
- [x] Country-risk enrichment prototype
- [x] Decision Tree, Random Forest and XGBoost baselines
- [ ] Refactor baseline into reproducible project structure
- [ ] Add temporal and currency-risk features
- [ ] Add behavioral/account-level features
- [ ] Build rule-based AML detection engine
- [ ] Add laundering-typology performance analysis
- [ ] Add network/graph features
- [ ] Compare rule-based and ML alerting
- [ ] Build interactive AML investigation dashboard
- [ ] Add tests and reproducible model outputs

## Repository structure

```text
.
├── data/
│   └── README.md
├── docs/
├── notebooks/
├── results/
│   └── figures/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

## Data

Large/raw datasets are intentionally excluded from version control. See `data/README.md` for the expected local data layout.

## Disclaimer

This is an educational/data-science portfolio project using simulated or research data. It is not a production AML system and should not be used to make real-world compliance decisions.
