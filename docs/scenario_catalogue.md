# AML Scenario Catalogue

The rule-based component is designed as a configurable, enterprise-style transaction-monitoring scenario engine. The implementations are original portfolio work based on common AML monitoring concepts and the observable fields in SAML-D; they do not reproduce confidential vendor or financial-institution configurations.

## Standard scenario design

Every scenario is specified through:

1. **Scenario ID and business rationale**
2. **Focal entity** — sender, receiver, or account/network
3. **Transaction population**
4. **Lookback window**
5. **Configurable thresholds**
6. **Aggregation / counterparty logic**
7. **Alert score and reason code**
8. **Evaluation metrics and typology coverage**

`Laundering_type` is ground truth for evaluation only and is never a scenario condition.

## Target catalogue — 18 scenarios

| ID | Family | Scenario | Core detection concept |
|---|---|---|---|
| SCN-001 | Amount | Single Large Transaction | One transaction exceeds a configurable high-value threshold |
| SCN-002 | Amount | Unusual Amount vs Historical Profile | Current amount materially exceeds focal account historical behavior |
| SCN-003 | Structuring | Repeated Sub-Threshold Activity | Multiple below-threshold transactions accumulate above an aggregate threshold |
| SCN-004 | Velocity | High Transaction Velocity | High transaction count within a rolling time window |
| SCN-005 | Velocity | Rapid Repeat Activity | Transactions recur within unusually short intervals |
| SCN-006 | Velocity | High Aggregate Value | Rolling aggregate value exceeds a configurable threshold |
| SCN-007 | Network | Fan-Out | One sender transacts with many distinct receivers within the lookback |
| SCN-008 | Network | Fan-In | One receiver receives from many distinct senders within the lookback |
| SCN-009 | Network | Gather-Scatter | Multiple inbound counterparties followed by multiple outbound counterparties |
| SCN-010 | Network | Scatter-Gather | Outbound dispersion followed by concentration into common destination(s) |
| SCN-011 | Flow of Funds | Rapid Movement / Pass-Through | Incoming funds are followed quickly by outgoing movement |
| SCN-012 | Flow of Funds | Circular Movement | Funds traverse a path that returns to an earlier account |
| SCN-013 | Network | Layered Fan-Out | Multi-hop outward dispersion across transaction layers |
| SCN-014 | Network | Layered Fan-In | Multi-hop inward concentration across transaction layers |
| SCN-015 | Geography | High-Risk Geography | Sender or receiver jurisdiction exceeds configurable country-risk threshold |
| SCN-016 | Geography | Sanctioned Geography | Sender or receiver is linked to a sanctioned jurisdiction indicator |
| SCN-017 | Geography / FX | Cross-Border Currency Mismatch | Cross-border movement combined with payment/received currency mismatch |
| SCN-018 | Behavioral | Behavioral Change | Material deviation from historical amount, velocity, geography, or counterparties |

## Alert output

Scenario hits are converted into a standardized alert layer:

```text
Alert_ID
Scenario_ID
Scenario_Name
Focal_Account
Alert_Timestamp
Lookback_Start
Lookback_End
Transaction_Count
Aggregate_Amount
Unique_Counterparties
Scenario_Score
Reason
```

This alert layer is the contract between detection and the future dashboard.

## SAML-D evaluation

The final evaluation will compare scenario coverage against all 17 SAML-D laundering typologies. Typology labels are revealed only after detection, allowing construction of an **18 scenarios × 17 laundering typologies** coverage matrix.

The system will report scenario-level alert volume, precision, recall, false positives, AML cases captured, and typology recall. Threshold tuning is performed on development/training data; final reported results are measured on an untouched holdout population.
