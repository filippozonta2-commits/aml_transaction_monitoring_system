# OPERATIONAL LAYER V1 — FREEZE

## Status
**FROZEN — downstream operationalization of Frozen Detection Portfolio V3**

This freeze governs alert aggregation, case generation, and case prioritization. It does **not** modify V3 scenario definitions, thresholds, DEVELOPMENT selection, or HOLDOUT validation results.

## End-to-end workload

| Layer | Volume |
|---|---:|
| HOLDOUT transactions | 1,898,009 |
| Unique flagged transactions | 127,981 |
| Transaction-level scenario triggers | 145,680 |
| Aggregated investigation alerts | 39,921 |
| Investigation cases | 24,804 |
| HIGH cases | 1,898 (7.65%) |
| MEDIUM cases | 9,651 (38.91%) |
| LOW cases | 13,255 (53.44%) |

Alert aggregation reduced scenario-trigger alert headers by **72.60%** while retaining all **127,981 unique evidence transactions** in ALERT_TRANSACTION.

## Detection boundary

Frozen V3 remains upstream and immutable under this operational freeze. HOLDOUT performance remains: **6.7429% flagged-transaction rate; 1,225/2,136 AML hits; 57.3502% recall; 0.9572% portfolio precision.** Operational aggregation/prioritization must never be represented as improving these metrics.

## Alert model

Raw scenario triggers are consolidated at **primary account × scenario × scenario-specific inactivity session**. Frozen windows: STRUCTURING 10d; DEPOSIT_SEND 3d; FAN_OUT 21d; FAN_IN 10d; CASH_WITHDRAWAL 7d; SMURFING 45d; UNUSUAL_AMOUNT_Z4 1d; SINGLE_LARGE_TRANSACTION 1d; GATHER_SCATTER 1d; HIGH_RISK_GEOGRAPHY 30d; SANCTIONED_GEOGRAPHY 30d.

The 30-day geography windows are workflow consolidation parameters, not detection thresholds. ALERT_TRANSACTION is the canonical transaction-evidence bridge.

## Case model

Aggregated alerts are sessionized by primary account. A new case starts when the gap from the previous alert exceeds **30 days**. Frozen output: 39,921 alerts; 24,804 cases; 1.61 alerts/case; 8,502 multi-alert cases (34.28%); 7,149 multi-scenario cases (28.82%); maximum 22 aggregated alerts/case. CASE_ALERT is canonical.

## Policy flag

policy_flag=True means at least one governed geography control (SCN_HIGH_RISK_GEOGRAPHY or SCN_SANCTIONED_GEOGRAPHY) is present. It does not establish money laundering or a SAR decision.

## Case prioritization

Transparent rule-based operational score using scenario severity, distinct scenarios, alert density, transaction volume, and governed-policy presence. **Is_laundering and Laundering_type are prohibited.** Bands: LOW 0–24; MEDIUM 25–49; HIGH 50–100. Policy-linked cases have a MEDIUM floor.

Frozen distribution: HIGH 1,898 (7.65%); MEDIUM 9,651 (38.91%); LOW 13,255 (53.44%). Audit: 98.21% of HIGH are multi-scenario; 90.99% policy-linked; only 34 HIGH are policy-only single-scenario; 1,693 HIGH combine policy with other scenarios; LOW contains no policy-linked cases.

## Governance constraints

1. Do not alter detection thresholds based on HOLDOUT outcomes.
2. Do not use HOLDOUT AML labels to tune aggregation or case priority.
3. Operational window/score changes require a new version and rationale.
4. Preserve transaction → alert → case lineage.
5. Preserve deterministic IDs where applicable.
6. Keep policy semantics distinct from empirical AML prediction.
7. Dashboards consume frozen outputs without silently recalculating logic.

## Canonical outputs

- results/case_management_v3/aggregated/alert.csv
- results/case_management_v3/aggregated/alert_transaction.csv
- results/case_management_v3/aggregated_cases/case.csv
- results/case_management_v3/aggregated_cases/case_alert.csv
- results/case_management_v3/prioritized/case_priority_queue.csv
- results/case_management_v3/prioritized/case_priority_evidence.csv

## Decision

**Operational Layer V1 is frozen.**

Future work may add investigator workflow, dashboarding, SLA metrics, disposition history, and reporting downstream of this freeze.