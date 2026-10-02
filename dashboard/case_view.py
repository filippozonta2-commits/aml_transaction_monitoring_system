import pandas as pd
import streamlit as st
import altair as alt

def render(d):
    q=d["queue"]; lin=d["lineage"]; net=d.get("network")
    st.header("Case Investigation")
    cid=st.selectbox("Case",q.CASE_ID.tolist())
    r=q[q.CASE_ID.eq(cid)].iloc[0]
    a,b,c,e=st.columns(4)
    a.metric("Priority rank",f"#{int(r.priority_rank):,}")
    b.metric("Risk score",f"{r.risk_score:.3f}")
    c.metric("Alerts",f"{int(r.transaction_alerts):,}")
    e.metric("Alert amount",f"${r.total_alert_amount:,.0f}")
    st.write("**Scenarios:**",r.scenario_list)

    ev=lin[lin.CASE_ID.eq(cid)].copy()
    if len(ev):
        ev["ts"]=pd.to_datetime(ev.ts)
        st.subheader("Alert timeline")
        st.scatter_chart(ev,x="ts",y="holdout_row_id",color="scenario")

    st.subheader("Transaction Network")
    if net is None:
        st.info("Run python src/build_case_network_data.py once to materialize the network view.")
    else:
        z=net[net.CASE_ID.eq(cid)].copy()
        if len(z):
            edges=(z.groupby(["holdout_row_id","Sender_account","Receiver_account"],dropna=False)
                     .agg(Amount=("Amount","first"),Payment_type=("Payment_type","first"),
                          sender_iso2=("sender_iso2","first"),receiver_iso2=("receiver_iso2","first"),
                          scenario=("scenario",lambda s:" | ".join(sorted(set(s.astype(str))))),
                          ts=("ts","first")).reset_index())
            chart=alt.Chart(edges).mark_circle(opacity=.75).encode(
                x=alt.X("Sender_account:N",title="Sender"),
                y=alt.Y("Receiver_account:N",title="Receiver"),
                size=alt.Size("Amount:Q",title="Amount"),
                color=alt.Color("scenario:N",title="Scenario"),
                tooltip=["holdout_row_id","Sender_account","Receiver_account","Amount","Payment_type",
                         "sender_iso2","receiver_iso2","scenario","ts"]
            ).properties(height=420).interactive()
            st.altair_chart(chart,use_container_width=True)
            st.caption("Each point is a real alerted transaction edge. Hover to inspect sender, receiver, amount, countries, payment type and scenario.")
            st.dataframe(edges.sort_values("Amount",ascending=False),use_container_width=True,hide_index=True)
        else:
            st.info("No transaction edges found for this case.")
    st.caption("The frozen ML score prioritizes human review; it is not an AML determination.")
