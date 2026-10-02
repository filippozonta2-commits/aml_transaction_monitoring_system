# AML Scenario Catalogue

The rule-based side of this project is designed as a configurable **scenario engine**, rather than a collection of hard-coded one-off rules.

The scenarios in this repository are original portfolio implementations based on common AML transaction-monitoring concepts and the fields available in SAML-D. They do **not** reproduce proprietary thresholds, source code, or confidential configurations from any vendor or financial institution.

## Scenario lifecycle

Each scenario should follow the same design:

1. **Business rationale** — what suspicious behavior is being targeted?
2. **Population** — which transactions/accounts are eligible?
3. **Lookback window** — what historical activity is considered?
4. **Parameters / thresholds** — configurable values, not hidden constants.
5. **Alert logic** — exact reproducible conditions.
6. **Evaluation** — alert volume, precision, recall, false positives, and typology recall.
7. **Tuning** — calibrate on training/reference data; evaluate once on holdout data.

## Phase 1 scenarios

| ID | Scenario | Status | Primary signal |
|---|---|---|---|
| SCN-001 | Single Large Transaction | Implemented | Transaction amount |
| SCN-002 | High-Risk Geography | Implemented | Country risk |
| SCN-003 | Sanctioned Geography | Implemented | Sanction indicator |
| SCN-004 | Cross-Border Currency Mismatch | Implemented | Geography + currency |
| SCN-005 | Rapid Repeat Activity | Implemented | Time since prior sender transaction |
| SCN-006 | High Velocity | Implemented | Prior sender activity |
| SCN-007 | Unusual Amount | Implemented | Amount vs sender history |
| SCN-008 | Structuring | Proxy | Repeated sub-threshold activity |
| SCN-009 | Fan Activity | Proxy | Repeated sender activity |

## Phase 2 network / rolling-window scenarios

The next feature layer will replace the proxy scenarios and add:

- true rolling-window structuring detection;
- fan-out using unique receivers within a time window;
- fan-in using unique senders within a time window;
- rapid movement / pass-through behavior;
- gather-scatter and scatter-gather patterns;
- circular/cycle activity;
- layered fan-in / fan-out;
- behavioral-change scenarios;
- counterparty concentration and network-risk signals.

## Ground-truth typologies

`Laundering_type` is reserved for evaluation and analysis. It must **not** be used as an input feature or scenario condition. This allows scenario recall to be measured by laundering typology without leaking the target into detection.
