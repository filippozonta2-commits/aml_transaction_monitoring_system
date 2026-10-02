import streamlit as st

def render(d):
    k=d["kpis"]; q=d["queue"]; sc=d["scenarios"]
    st.header("Executive Overview")
    cols=st.columns(5)
    vals=[("Transactions",f"{int(k.transactions):,}"),("Transaction alerts",f"{int(k.transaction_alerts):,}"),
          ("Alert rate",f"{k.transaction_alert_rate:.2%}"),("Cases",f"{int(k.cases):,}"),("Accounts",f"{int(k.accounts):,}")]
    for c,(n,v) in zip(cols,vals): c.metric(n,v)
    st.subheader("Frozen scenario volumes")
    st.bar_chart(sc.set_index("scenario")["triggered_transactions"].sort_values(),horizontal=True)
    st.subheader("Investigator priority distribution")
    st.bar_chart(q["priority_band"].value_counts().reindex(["Critical","High","Medium","Standard"]).fillna(0))
