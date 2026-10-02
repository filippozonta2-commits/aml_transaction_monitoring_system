# AML Scenario Portfolio — DEVELOPMENT Freeze Manifest

Freeze date: 2026-10-02
Status: FROZEN BEFORE HOLDOUT
Portfolio version: v3.0-development-freeze

## Governance rule

All scenario definitions and thresholds below were selected using TRAIN warm-up
and/or DEVELOPMENT only. After this manifest is committed, HOLDOUT may be used
for final evaluation, but no threshold or scenario definition may be changed
based on HOLDOUT performance. Any later change requires a new portfolio version
and a fresh untouched evaluation set.

## Frozen portfolio (11 controls)

1. SCN_STRUCTURING
   - 10-day receiver fixed buckets.
   - Amount < 10,000; receiver unique senders >= 5; receiver aggregate >= 20,000.
   - Cross-border rate >= 0.20 OR currency-mismatch rate >= 0.25.

2. SCN_DEPOSIT_SEND
   - 72-hour forward receiver-flow rule.
   - outgoing_count >= 1; outflow_ratio in [0.50, 3.00]; cross_border_count >= 1.
   - TRAIN used only as 72-hour warm-up context.

3. SCN_FAN_OUT
   - 21-day sender fixed windows.
   - unique receivers >= 3; transaction count 3–12.
   - cross-border rate >= 0.20 OR currency-mismatch rate >= 0.20.

4. SCN_FAN_IN
   - 10-day receiver fixed buckets.
   - unique senders 5–15; receiver transaction count 5–20.
   - cross-border rate >= 0.10 OR currency-mismatch rate >= 0.20.

5. SCN_CASH_WITHDRAWAL
   - Cash Withdrawal only; 7-day sender fixed windows.
   - transaction count >= 5; aggregate amount >= 300; median amount <= 300.

6. SCN_SMURFING
   - Cash Deposit only; 45-day sender fixed windows.
   - transaction count >= 3; median amount < 4,000; aggregate amount >= 10,000.
   - cross-border rate <= 0.15; currency-mismatch rate <= 0.15.

7. SCN_UNUSUAL_AMOUNT_Z4
   - Prior sender history only.
   - At least 10 prior transactions; prior-history standard deviation > 0.
   - current amount z-score >= 4 relative to prior sender history.

8. SCN_SINGLE_LARGE_TRANSACTION
   - Existing DEVELOPMENT candidate definition/materialization is frozen as used
     in candidate_flags_development.csv at this portfolio freeze.

9. SCN_GATHER_SCATTER
   - Existing DEVELOPMENT candidate definition/materialization is frozen as used
     in candidate_flags_development.csv at this portfolio freeze.

10. SCN_HIGH_RISK_GEOGRAPHY
    - Governed country-risk registry.
    - HIGH when project country-risk Overall Score >= 6.50.
    - Policy-driven control; empirical uniqueness is not the sole retention test.

11. SCN_SANCTIONED_GEOGRAPHY
    - Governed sanctions flag from project country-risk reference.
    - Policy-driven control; empirical uniqueness is not the sole retention test.

## Explicitly excluded from this freeze

- SCN_CROSS_BORDER_CURRENCY_MISMATCH_Z4: excluded because final DEVELOPMENT
  ablation showed 0 unique portfolio alerts and 0 unique AML hits.
- SCN_HIGH_TRANSACTION_VELOCITY: redesign required.
- SCN_BEHAVIORAL_CHANGE: redesign required.
- SCN_SCATTER_GATHER, SCN_CIRCULAR_MOVEMENT, SCN_LAYERED_FAN_OUT,
  SCN_LAYERED_FAN_IN: zero-trigger DEVELOPMENT candidates; not frozen.

## DEVELOPMENT freeze metrics

Transactions: 1,899,527
AML positives: 1,986
Frozen 11-control portfolio alerts: 131,921
Alert rate: 6.9449%
AML hits: 1,189
AML recall: 59.8691%

The excluded cross-border mismatch Z4 control had zero unique alerts in the
12-control ablation, so removing it leaves these portfolio-level metrics unchanged.

## Final DEVELOPMENT ablation evidence

Unique AML hits lost if each retained control is removed:
- SCN_SMURFING: 193
- SCN_CASH_WITHDRAWAL: 168
- SCN_FAN_OUT: 158
- SCN_FAN_IN: 81
- SCN_DEPOSIT_SEND: 72
- SCN_SANCTIONED_GEOGRAPHY: 54
- SCN_SINGLE_LARGE_TRANSACTION: 34
- SCN_GATHER_SCATTER: 33
- SCN_UNUSUAL_AMOUNT_Z4: 24
- SCN_HIGH_RISK_GEOGRAPHY: 22
- SCN_STRUCTURING: 0

SCN_STRUCTURING remains frozen because it is an established typology-specific
control; portfolio ablation redundancy alone does not invalidate its scenario
purpose. Geography controls are additionally retained for governed policy coverage.

## HOLDOUT protocol

1. Do not tune, add, remove, or reinterpret controls after viewing HOLDOUT.
2. Materialize these exact 11 definitions on HOLDOUT.
3. Report alert rate, AML precision/recall, typology coverage, overlap, and
   portfolio-level metrics.
4. Compare HOLDOUT with DEVELOPMENT as an out-of-sample validation, not as a
   new tuning cycle.
5. Record any failure or degradation as a result. Do not repair it using HOLDOUT.
