"""Candidate expansion from 6 frozen scenarios to the agreed 18-scenario catalogue.

These 12 scenarios are DEVELOPMENT candidates only. They are deliberately kept
out of scenario_config_frozen.py and the HOLDOUT pipeline until validated.
"""
CANDIDATE_SCENARIOS={
"SCN_SINGLE_LARGE_TRANSACTION":{"family":"Amount","status":"candidate_development","definition":"Single transaction unusually large relative to DEVELOPMENT amount distribution."},
"SCN_UNUSUAL_AMOUNT":{"family":"Amount","status":"candidate_development","definition":"Transaction amount unusually large relative to the sender's own historical profile."},
"SCN_HIGH_TRANSACTION_VELOCITY":{"family":"Velocity","status":"candidate_development","definition":"Unusually high transaction count for an account in a short rolling/fixed window."},
"SCN_GATHER_SCATTER":{"family":"Network","status":"candidate_development","definition":"Account receives from multiple senders and redistributes to multiple receivers in a short window."},
"SCN_SCATTER_GATHER":{"family":"Network","status":"candidate_development","definition":"Funds scatter across intermediaries and reconverge on a common downstream account."},
"SCN_CIRCULAR_MOVEMENT":{"family":"Flow of Funds","status":"candidate_development","definition":"Directed transaction cycle involving three or more accounts."},
"SCN_LAYERED_FAN_OUT":{"family":"Flow of Funds","status":"candidate_development","definition":"Fan-out followed by another outward layer through recipient accounts."},
"SCN_LAYERED_FAN_IN":{"family":"Flow of Funds","status":"candidate_development","definition":"Multiple upstream branches converge through intermediaries toward a common account."},
"SCN_HIGH_RISK_GEOGRAPHY":{"family":"Geography / FX","status":"candidate_development","definition":"Transaction involving a country classified as high risk in the project country-risk table."},
"SCN_SANCTIONED_GEOGRAPHY":{"family":"Geography / FX","status":"candidate_development","definition":"Transaction involving a geography marked sanctioned/restricted in the project reference data."},
"SCN_CROSS_BORDER_CURRENCY_MISMATCH":{"family":"Geography / FX","status":"candidate_development","definition":"Cross-border transfer where payment and received currencies differ."},
"SCN_BEHAVIORAL_CHANGE":{"family":"Behavioral","status":"candidate_development","definition":"Material deviation from an account's prior transaction amount/velocity profile."},
}

CATALOGUE_MAPPING={
"SCN_SMURFING":"Repeated Sub-Threshold Activity",
"SCN_CASH_WITHDRAWAL":"Rapid Repeat Activity",
"SCN_FAN_OUT":"Fan-Out",
"SCN_FAN_IN":"Fan-In",
"SCN_DEPOSIT_SEND":"Rapid Movement / Pass-Through",
"SCN_STRUCTURING":"High Aggregate Value / Structuring",
}
