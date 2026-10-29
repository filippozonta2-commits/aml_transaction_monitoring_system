# AML Transaction Monitoring & Investigation Console

An end-to-end **educational AML analytics portfolio project** built on simulated SAML-D transactions. The repository contains chronological transaction-monitoring experiments, a frozen rule-based investigation workflow, an experimental XGBoost detector, and a Streamlit investigation console with case networks and geographic context.

> **Important version distinction:** The restored investigation console uses a previously frozen **11-scenario Rule-Based V3 / Operational Layer V1** pipeline and its associated case-management artifacts. The separately documented **four-scenario rolling-window prototype** in `src/scenario_engine_v3.py` is **not** the same detector or evaluation. Do not compare their results as if they were one model.

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

## Frozen ML Detection V1 → operational cases (locally verified)

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

These figures were observed from a local HOLDOUT run and checked in the Streamlit ML Case Management page. The workflow preserves links to all **39,409** flagged transactions. The operational case count is **23.59% lower** than the transaction-alert count; this is **not** a measured reduction in investigation hours, nor an improvement in detection recall.

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

## Separate experimental four-scenario engine

The causal rolling-window prototype detects candidate Smurfing, Fan-Out, Structuring-proxy and Fan-In patterns. It evaluates DEVELOPMENT with TRAIN as historical warm-up, and HOLDOUT with frozen thresholds. This prototype is **separate from the 11-scenario dashboard baseline**.

| Metric | DEVELOPMENT | HOLDOUT |
|---|---:|---:|
| Transactions | 1,899,527 | 1,898,009 |
| Labeled AML transactions | 1,986 | 2,136 |
| Deduplicated alerts | 66,215 | 66,012 |
| Labeled AML hits | 503 | 525 |
| Recall | 25.33% | 24.58% |
| Precision | 0.76% | 0.80% |
| Alert rate | 3.49% | 3.48% |

Reproduce with the temporal splits described in `data/README.md`:

```bash
python3 -m unittest discover -s tests -v
python3 -m src.scenario_engine_v3
python3 -m src.evaluate_holdout_v3
```

The low precision and recall are disclosed intentionally; these results are not evidence of deployable AML effectiveness. The structuring proxy is **not** a legal cash-reporting threshold test.

## Evaluation and governance

The console compares its **frozen 11-scenario Rule-Based V3** with the **frozen transaction-level XGBoost Detection V1** only where matching HOLDOUT artifacts are present. The ML recall and precision displayed there are computed from saved row-level labels and predictions, not hardcoded. The separate case-prioritization ML model is a different component from transaction-level detection.

Chronological splits, training/development model selection and frozen HOLDOUT evaluation are intended to reduce look-ahead bias. HOLDOUT should not be used to tune thresholds. The simulation lab is intentionally disabled for legacy scenario configurations.

See `docs/bsa_aml_regulatory_alignment.md` for regulatory interpretation and limitations.

## Technology

Python, pandas, NumPy, scikit-learn, XGBoost, Streamlit, Plotly, NetworkX, Matplotlib and country normalization.

## Disclaimer

**Research/portfolio prototype using simulated transactions.** Alert flags and geographic patterns are risk indicators, not determinations of money laundering, suspicious activity reports or regulatory compliance decisions. Human review is required. The repository is not a production transaction-monitoring system.
