# AML Transaction Monitoring & Investigation Console

An end-to-end **educational AML analytics portfolio project** built on simulated SAML-D transactions. The repository documents a frozen 11-scenario rule-based detection and investigation workflow, a frozen transaction-level XGBoost detector with independent ML case management, and a Streamlit investigation console with network and geographic context.

## Independent validation status (2026-10-29)

The frozen HOLDOUT metrics below are **previously reported project results**, not newly reproduced by the independent repository review. An offline audit has reconciled the published confusion-matrix arithmetic, but the public checkout does **not** include the frozen XGBoost training/scoring implementation, model artifact, exact feature schema, DEVELOPMENT threshold-selection log or full HOLDOUT score/label file. Therefore the reported **84.18% recall is not yet independently verified**, and PR-AUC cannot be calculated from the published summary counts.

The legacy exploratory `src/baseline_models.py` uses a random stratified split after feature engineering and **must not be treated as the frozen chronological model**. The generic `src/rule_based.py` currently lists **12** scenario keys, so it has **not** been established as the exact implementation of the reported frozen **11-scenario** Rule-Based V3. The country-level supervised target-rate features mentioned in earlier portfolio descriptions have no auditable source/provenance in this repository; their leakage status remains **unresolved**. See [Independent validation audit](docs/independent_validation_audit_2026-10-29.md) and [Model validation & reproducibility gates](docs/model_validation_gate.md) and run `python -m src.audit_frozen_metrics` for summary arithmetic reconciliation. Row-level audit requires real joined HOLDOUT evidence.

## Investigation console

Launch from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

The console provides:
- Monitoring Overview and frozen Rule-Based V3 vs ML Detection V1 comparison
- Scenario Monitoring and an investigator-prioritized case queue
- Case Investigation: alert lineage, transaction evidence, account network and country-level geography
- ML analytics, independent ML investigator queue, ML case investigation and governance/lineage views

The console loads **locally generated CSV artifacts** from `results/`. Raw data and large intermediate outputs are not committed. A working checkout of the code alone does not guarantee that every console page has its required artifacts.

### Required console artifacts

- `results/case_management_v3/prioritized/case_priority_queue.csv`
- `results/case_management_v3/aggregated/alert.csv`
- `results/case_management_v3/aggregated/alert_transaction.csv`
- `results/case_management_v3/aggregated_cases/case_alert.csv`
- `results/holdout_v3/final_metrics.csv`

Optional, feature-specific artifacts:
- `results/dashboard/case_transaction_network_v3.csv` — network and geography
- `results/ml_case_v3/ml_case_scores.csv` — legacy rule-case ML prioritization (distinct from ML detection and independent ML cases)
- `results/ml_case_v3/ml_feature_importance.csv`
- `results/ml_detection/holdout_v1/ml_holdout_scores.csv` — transaction-level frozen ML detector
- `results/comparison_rule_v3_vs_ml_v1/overlap_summary.csv` — frozen detector overlap
- `results/ml_detection/full_train_development/operating_points.csv`

To rebuild the case-level geographic/network context **after the original V3/V1 case-management outputs exist**:

```bash
python3 -m src.build_case_network_data
```

This uses `data/temporal/SAML-D_holdout.csv` and creates `results/dashboard/case_transaction_network_v3.csv`. The geography tab displays observed country-level transaction context, not verified physical locations or evidence of wrongdoing.

## Reported frozen HOLDOUT detection comparison (not independently reproduced)

The **11-scenario Rule-Based V3** and **XGBoost Detection V1** were evaluated on the same chronological HOLDOUT population of **1,898,009 transactions**, including **2,136 labeled AML positives**. The figures below are *transaction-level detection* results, not aggregated alert or case counts.

| Metric | Rule-Based V3 (11 scenarios) | XGBoost Detection V1 |
|---|---:|---:|
| Flagged transactions | 127,981 | 39,409 |
| Transaction alert rate | 6.74% | 2.08% |
| True-positive AML transactions | 1,225 | 1,798 |
| Transaction precision | 0.96% | **4.56%** |
| Transaction recall | 57.35% | **84.18%** |

According to the reported frozen benchmark, XGBoost identifies **573 additional AML-positive transactions**, improves recall by **26.83 percentage points**, and flags **69.21% fewer transactions**. These are aggregate detector-level comparisons; they do not establish how many of the same individual positives both systems detect. Detection precision remains low in absolute terms because of the strongly imbalanced simulated population.

**Different grains:** the Rule-Based count of 127,981 and ML count of 39,409 are flagged *transactions*. The ML operational workflow subsequently produces 35,107 *aggregated alerts* and 30,114 *cases*. Comparing these counts without distinguishing their units would be misleading.

## Reported frozen ML Detection V1 → operational cases (not independently reproduced)

The frozen transaction-level XGBoost model is evaluated **before** operational aggregation. The downstream process does not retrain the model, change its decision threshold, or use AML ground-truth labels for case generation.

```text
HOLDOUT transactions
  → frozen XGBoost probability and detection threshold
  → 39,409 flagged transactions
  → sender-account + 7-day inactivity-session alert aggregation
  → 35,107 aggregated ML alerts
  → sender-account + 30-day inactivity-session case generation
  → 30,114 independent ML investigation cases
  → ML Investigator Queue → Human Review
```

These figures were reported from a prior local HOLDOUT run and described as checked in the Streamlit ML Case Management page; this public repository review did not reproduce that run. The workflow preserves links to all **39,409** flagged transactions. The operational case count is **23.59% lower** than the transaction-alert count; this is **not** a measured reduction in investigation hours, nor an improvement in detection recall.

| Stage | Count | Reduction from previous stage |
|---|---:|---:|
| Frozen XGBoost transaction alerts | 39,409 | — |
| Aggregated ML investigation alerts | 35,107 | 10.92% |
| Independent ML investigation cases | 30,114 | 14.22% |

**Aggregation choice:** the subject is the **sender account**, not the recipient or a customer-level multi-account entity. Multiple flagged transactions from one sender can share an alert if their successive timestamps are within seven days; aggregated alerts can share a case if their successive start times are within 30 days. These are *inactivity sessions*, not calendar buckets. Other entity-resolution strategies could produce different case volumes.

With the frozen ML score artifact and original temporal HOLDOUT CSV available locally, rebuild the ML operational layer using:

```bash
python3 -m src.build_ml_aggregated_alert_layer_v1
python3 -m src.build_ml_case_management_v1
python3 -m streamlit run app.py
```

Outputs are written under `results/ml_case_management_v1/aggregated/` and `results/ml_case_management_v1/cases/`. The ML case queue ranks cases by their **maximum triggering transaction probability**; this is an operational ordering heuristic, **not a calibrated case-level risk probability**.

### Rule-Based V3 operational workflow

The independent frozen **11-scenario Rule-Based V3** produces transaction-level scenario triggers, then groups them by **sender account + scenario + scenario-specific inactivity session**. Its aggregated alerts are grouped into cases by **account + 30-day inactivity session**, followed by operational prioritization and human investigation. Unlike ML, rule-based aggregation retains scenario separation at the alert level. The two pipelines must not share case counts or imply equivalent case definitions.

Governance & Lineage displays each pipeline separately, using locally loaded artifact counts and checking alert-to-transaction and case-to-alert referential integrity.

## Evaluation and governance

The console compares its **frozen 11-scenario Rule-Based V3** with the **frozen transaction-level XGBoost Detection V1** only where matching HOLDOUT artifacts are present. The ML recall and precision displayed there are computed from saved row-level labels and predictions, not hardcoded. The separate case-prioritization ML model is a different component from transaction-level detection.

Chronological splits, training/development model selection and frozen HOLDOUT evaluation are intended to reduce look-ahead bias. HOLDOUT should not be used to tune thresholds. The simulation lab is intentionally disabled for legacy scenario configurations.

See `docs/bsa_aml_regulatory_alignment.md` for regulatory interpretation and limitations.

## Technology

Python, pandas, NumPy, scikit-learn, XGBoost, Streamlit, Plotly, NetworkX, Matplotlib and country normalization.

## Disclaimer

**Research/portfolio prototype using simulated transactions.** Alert flags and geographic patterns are risk indicators, not determinations of money laundering, suspicious activity reports or regulatory compliance decisions. Human review is required. The repository is not a production transaction-monitoring system.
