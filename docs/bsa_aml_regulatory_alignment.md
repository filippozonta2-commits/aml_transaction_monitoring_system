# BSA/AML and USA PATRIOT Act alignment — V3 scenario design

**Scope:** Educational transaction-monitoring prototype using simulated SAML-D transfers. These rules are **risk indicators**, not mandated legal tests, regulatory compliance certification, or automatic SAR decisions. Current frozen HOLDOUT results are retained without modification.

## Regulatory framework

- **BSA / FinCEN:** Detect, investigate and report suspicious activity as applicable. Suspicion depends on context; an alert alone does not establish reportability.
- **USA PATRIOT Act:** Section 326 informs Customer Identification Program (CIP); Section 314 concerns information sharing. Neither establishes numerical Smurfing/Fan-In/Fan-Out detection thresholds. CIP/CDD and 314 workflows cannot be implemented solely from transaction rows.
- **FFIEC BSA/AML Examination Manual:** Risk-based monitoring, documented scenarios and thresholds, alert management, escalation, independent testing, periodic tuning and consideration of customer/product/geographic risk.

## Scenario-to-risk mapping

| Scenario | Relevant suspicious behavior | V3 implementation | Compliance caveat |
|---|---|---|---|
| Smurfing | Repeated small transfers that may conceal source or fragment activity | 90-day sender rolling count, median, sum and receiver concentration | A $3,000 median is a **tuned model parameter**, not a statutory threshold. Transfers alone do not establish cash structuring. |
| Structuring | Deliberate splitting of **currency/cash** transactions to evade CTR obligations | Current 10-day receiver aggregation, sub-$10,000 individual amount, multiple senders, geography/FX | **Not a valid CTR-evasion detector without cash-transaction type, USD-equivalent amounts, same-person linkage, and relevant timing.** Rename analytically as `STRUCTURING_PROXY` in reports until those data exist. The $10,000 CTR rule is about qualifying currency transactions; do not imply it applies to all electronic transfers. |
| Fan-Out | Rapid dispersion of funds to multiple beneficiaries, potentially layering | 21-day sender rolling counterparties, count and geographic/FX signals | Typology indicator; no universal statutory counterparty cutoff. |
| Fan-In | Multiple funding sources concentrating funds into a recipient, potentially layering | 10-day receiver rolling counterparties, count and geographic/FX signals | Typology indicator; no universal statutory cutoff. |

## Controls required for closer bank-style monitoring

1. **Data dictionary and product segmentation:** transaction channel, cash/wire/ACH, normalized USD amount, customer/account mapping, counterparty, geography, expected activity.
2. **CIP/CDD context:** customer risk rating, occupation/business, source of funds, expected activity and beneficial ownership where applicable. Unavailable fields must be marked missing, never fabricated.
3. **Scenario governance:** rule ID/version, rationale, time window, configurable thresholds, change approval, regression tests, periodic back-testing, and independent validation.
4. **Case workflow:** alert ID, account/customer, trigger reason, supporting transactions, analyst disposition, escalation, investigation notes, and potential SAR referral. Filing decisions require authorized human review.
5. **CTR and SAR distinction:** CTR generally covers qualifying cash transactions over $10,000 in one business day (with applicable aggregation). A bank SAR may be required at $5,000 or more when relevant suspicion conditions apply; that is **not** an automatic alert threshold or blanket safe harbor below $5,000.
6. **Holdout discipline:** Do not revise existing V3 parameters after observing HOLDOUT. Any materially revised rules are **V4**, with a new prospective/untouched evaluation period needed for an unbiased final estimate.

## Current performance (frozen V3)

HOLDOUT: 1,898,009 transactions; 2,136 labeled AML; 66,012 alerts; 525 AML hits; **24.58% recall**, **0.80% precision**, **3.48% alert rate**. The dataset's labels are synthetic and not bank SAR outcomes.

## Authoritative references

- FFIEC BSA/AML Examination Manual — Suspicious Activity Reporting: https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements/04
- FFIEC BSA/AML Risk Assessment: https://bsaaml.ffiec.gov/manual/BSAAMLRiskAssessment/01
- FinCEN, Suspicious Activity Reporting (Structuring): https://www.fincen.gov/resources/statutes-regulations/administrative-rulings/suspicious-activity-reporting-structuring
- NCUA/Interagency SAR FAQs on structuring: https://ncua.gov/regulation-supervision/regulatory-compliance-resources/bank-secrecy-act-resources/frequently-asked-questions-regarding-suspicious-activity-reporting
- FFIEC manual index (CIP, CDD, CTR, SAR and related controls): https://bsaaml.ffiec.gov/manual
