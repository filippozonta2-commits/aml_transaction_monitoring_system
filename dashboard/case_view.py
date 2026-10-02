import pandas as pd
import streamlit as st
import altair as alt
import networkx as nx
import matplotlib.pyplot as plt

def _edges(z):
    return (z.groupby(["holdout_row_id","Sender_account","Receiver_account"],dropna=False)
             .agg(Amount=("Amount","first"),Payment_type=("Payment_type","first"),
                  sender_iso2=("sender_iso2","first"),receiver_iso2=("receiver_iso2","first"),
                  scenario=("scenario",lambda s:" | ".join(sorted(set(s.astype(str))))),
                  ts=("ts","first"),cross_border=("cross_border","first"))
             .reset_index())

def _network(edges,account):
    G=nx.DiGraph()
    for _,x in edges.iterrows():
        u=str(x.Sender_account); v=str(x.Receiver_account)
        if G.has_edge(u,v):
            G[u][v]["amount"]+=float(x.Amount); G[u][v]["count"]+=1
        else: G.add_edge(u,v,amount=float(x.Amount),count=1)
    fig,ax=plt.subplots(figsize=(12,7))
    pos=nx.spring_layout(G,seed=42,k=max(0.5,2/(max(len(G),1)**0.5)))
    sizes=[1400 if n==str(account) else 500 for n in G.nodes]
    widths=[0.7+min(4,G[u][v]["count"]*.35) for u,v in G.edges]
    nx.draw_networkx_nodes(G,pos,node_size=sizes,ax=ax)
    nx.draw_networkx_edges(G,pos,width=widths,arrows=True,arrowsize=16,alpha=.55,ax=ax)
    nx.draw_networkx_labels(G,pos,font_size=7,ax=ax)
    ax.axis("off"); return fig

def render(d):
    q=d["queue"]; lin=d["lineage"]; net=d.get("network")
    st.header("Case Investigation")
    cid=st.selectbox("Case",q.CASE_ID.tolist())
    r=q[q.CASE_ID.eq(cid)].iloc[0]
    a,b,c,e=st.columns(4)
    a.metric("Priority rank",f"#{int(r.priority_rank):,}"); b.metric("Risk score",f"{r.risk_score:.3f}")
    c.metric("Alerts",f"{int(r.transaction_alerts):,}"); e.metric("Alert amount",f"${r.total_alert_amount:,.0f}")
    st.write("**Scenarios:**",r.scenario_list)
    ev=lin[lin.CASE_ID.eq(cid)].copy()
    if len(ev):
        ev["ts"]=pd.to_datetime(ev.ts); st.subheader("Alert timeline")
        st.scatter_chart(ev,x="ts",y="holdout_row_id",color="scenario")

    st.subheader("Transaction Network")
    if net is None: st.info("Run python src/build_case_network_data.py once.")
    else:
        z=net[net.CASE_ID.eq(cid)].copy()
        if len(z):
            edges=_edges(z)
            m1,m2,m3=st.columns(3)
            m1.metric("Counterparties",len(set(edges.Sender_account.astype(str))|set(edges.Receiver_account.astype(str)))-1)
            m2.metric("Cross-border tx",int(edges.cross_border.sum()))
            m3.metric("Countries",len(set(edges.sender_iso2.dropna())|set(edges.receiver_iso2.dropna())))
            st.pyplot(_network(edges,r.account_id),width="stretch")
            st.caption("Directed account graph. Arrows show money flow; thicker links represent repeated alerted transfers.")
            st.subheader("Geographic flows")
            geo=(edges.groupby(["sender_iso2","receiver_iso2"],dropna=False)
                 .agg(transactions=("holdout_row_id","count"),amount=("Amount","sum")).reset_index())
            geo["route"]=geo.sender_iso2.astype(str)+" → "+geo.receiver_iso2.astype(str)
            chart=alt.Chart(geo).mark_bar().encode(
                x=alt.X("amount:Q",title="Alerted amount"),y=alt.Y("route:N",sort="-x",title="Country route"),
                tooltip=["sender_iso2","receiver_iso2","transactions","amount"]).properties(height=min(500,max(180,28*len(geo))))
            st.altair_chart(chart,width="stretch")
            with st.expander("Transaction details"):
                st.dataframe(edges.sort_values("Amount",ascending=False),width="stretch",hide_index=True)
        else: st.info("No transaction edges found for this case.")
    st.caption("The frozen ML score prioritizes human review; it is not an AML determination.")
