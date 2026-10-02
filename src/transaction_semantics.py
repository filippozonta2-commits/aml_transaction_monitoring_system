"""Canonical semantic interpretation of the uniform SAML-D transaction schema.

Raw Sender_account / Receiver_account are preserved. Derived flags define when
two-sided counterparty/geography/currency semantics are valid. Scenario, network,
ML and dashboard code should consume these flags rather than reinterpret cash
transactions independently.
"""
import numpy as np

def add_transaction_semantics(df):
    x=df.copy()
    pt=x["Payment_type"].fillna("").astype(str).str.strip()
    wd=pt.eq("Cash Withdrawal")
    dep=pt.eq("Cash Deposit")
    cash=wd|dep

    x["is_cash_withdrawal"]=wd.astype("int8")
    x["is_cash_deposit"]=dep.astype("int8")
    x["is_cash_transaction"]=cash.astype("int8")
    x["is_account_transfer"]=(~cash).astype("int8")
    x["is_cross_border_transfer"]=(pt.eq("Cross-border") & ~cash).astype("int8")

    x["flow_type"]=np.select(
        [wd,dep],["CASH_WITHDRAWAL","CASH_DEPOSIT"],default="ACCOUNT_TRANSFER"
    )
    x["initiating_account"]=np.where(
        wd,x["Sender_account"],np.where(dep,x["Receiver_account"],x["Sender_account"])
    )
    x["counterparty_account"]=np.where(cash,np.nan,x["Receiver_account"])

    # Semantic applicability flags. Raw fields remain untouched for lineage.
    x["has_true_sender_receiver"]=(~cash).astype("int8")
    x["counterparty_applicable"]=(~cash).astype("int8")
    x["geo_pair_applicable"]=(~cash).astype("int8")
    x["currency_pair_applicable"]=(~cash).astype("int8")
    x["network_eligible"]=(~cash).astype("int8")

    x["sender_is_customer"]=(~dep).astype("int8")
    x["receiver_is_customer"]=(~wd).astype("int8")
    x["transaction_direction"]=np.select(
        [wd,dep],["CASH_OUT","CASH_IN"],default="OUTBOUND"
    )
    x["semantic_direction"]=np.select(
        [wd,dep],["ACCOUNT → CASH","CASH → ACCOUNT"],default="SENDER → RECEIVER"
    )
    return x
