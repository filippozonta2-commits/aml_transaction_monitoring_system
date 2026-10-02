"""Semantic interpretation of the uniform SAML-D transaction schema.

Raw Sender_account / Receiver_account are preserved. These helpers add an
investigation-facing meaning without changing frozen scenario/model inputs.
"""
import numpy as np

def add_transaction_semantics(df):
    x=df.copy()
    pt=x["Payment_type"].fillna("").astype(str)
    wd=pt.eq("Cash Withdrawal"); dep=pt.eq("Cash Deposit")
    x["flow_type"]=np.select([wd,dep],["CASH_WITHDRAWAL","CASH_DEPOSIT"],default="ACCOUNT_TRANSFER")
    x["initiating_account"]=np.where(wd,x["Sender_account"],np.where(dep,x["Receiver_account"],x["Sender_account"]))
    x["counterparty_account"]=np.where(wd|dep,np.nan,x["Receiver_account"])
    x["network_eligible"]=(~(wd|dep)).astype("int8")
    x["semantic_direction"]=np.select([wd,dep],["ACCOUNT → CASH","CASH → ACCOUNT"],default="SENDER → RECEIVER")
    return x
