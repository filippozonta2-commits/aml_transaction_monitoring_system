import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pycountry

def _edges(z):
    return (z.groupby(["holdout_row_id","Sender_account","Receiver_account"],dropna=False)
             .agg(Amount=("Amount","first"),Payment_type=("Payment_type","first"),
                  sender_iso2=("sender_iso2","first"),receiver_iso2=("receiver_iso2","first"),
                  scenario=("scenario",lambda s:" | ".join(sorted(set(s.astype(str))))),
                  ts=("ts","first"),cross_border=("cross_border","first")).reset_index())

def _network(edges,account):
    G=nx.DiGraph()
    for _,x in edges.iterrows():
        u=str(x.Sender_account); v=str(x.Receiver_account)
        if G.has_edge(u,v): G[u][v]["count"]+=1
        else: G.add_edge(u,v,count=1)
    fig,ax=plt.subplots(figsize=(11,6))
    pos=nx.spring_layout(G,seed=42,k=max(.5,2/(max(len(G),1)**.5)))
    nx.draw_networkx_nodes(G,pos,node_size=[1500 if n==str(account) else 430 for n in G],ax=ax)
    nx.draw_networkx_edges(G,pos,width=[.8+min(4,G[u][v]["count"]*.35) for u,v in G.edges],arrows=True,arrowsize=16,alpha=.55,ax=ax)
    nx.draw_networkx_labels(G,pos,font_size=7,ax=ax); ax.axis("off"); return fig

def _iso3(x):
    try:return pycountry.countries.get(alpha_2=str(x)).alpha_3
    except:return None

def _world_map(edges):
    s=edges.groupby("sender_iso2").agg(sent=("Amount","sum"),outgoing=("holdout_row_id","count"))
    r=edges.groupby("receiver_iso2").agg(received=("Amount","sum"),incoming=("holdout_row_id","count"))
    g=s.join(r,how="outer").fillna(0).reset_index().rename(columns={"index":"iso2","sender_iso2":"iso2","receiver_iso2":"iso2"})
    g["iso3"]=g.iso2.map(_iso3); g["total_amount"]=g.sent+g.received; g["transactions"]=g.outgoing+g.incoming
    fig=px.choropleth(g,locations="iso3",color="total_amount",hover_name="iso2",
                      hover_data={"iso3":False,"sent":":,.0f","received":":,.0f","transactions":":,.0f"},
                      projection="natural earth",labels={"total_amount":"Alerted amount"})
    fig.update_geos(showcoastlines=True,showland=True,fitbounds="locations")
    fig.update_layout(height=500,margin=dict(l=0,r=0,t=10,b=0),coloraxis_colorbar_title="Amount")
    return fig,g

def render(d):
    q=d["queue"]; lin=d["lineage"]; net=d.get("network")
    st.header("Case Investigation")
    cid=st.selectbox("Select investigation case",q.CASE_ID.tolist())
    r=q[q.CASE_ID.eq(cid)].iloc[0]
    st.markdown(f"### {cid}  ·  Account {r.account_id}")
    st.markdown("#### Detection — Scenario Engine")
    st.caption("Why was this case created? Frozen rule-based scenarios generated the underlying transaction alerts.")
    a,b,c,e=st.columns(4)
    a.metric("Priority",f"#{int(r.priority_rank):,} · {r.priority_band}"); b.metric("ML risk score",f"{r.risk_score:.3f}")
    c.metric("Alert events",f"{int(r.transaction_alerts):,}"); e.metric("Alerted amount",f"${r.total_alert_amount:,.0f}")
    st.caption(f"Active scenarios: {r.scenario_list}")
    st.markdown("#### Prioritization — Machine Learning")
    st.caption("Why should this case be reviewed earlier? The frozen XGBoost model ranks cases after case creation; it does not generate alerts.")
    a,b=st.columns(2); a.metric("Priority",f"#{int(r.priority_rank):,} · {r.priority_band}"); b.metric("ML risk score",f"{r.risk_score:.3f}")
    if net is None:
        st.info("Run python src/build_case_network_data.py once."); return
    z=net[net.CASE_ID.eq(cid)].copy()
    if not len(z): st.info("No transaction edges found for this case."); return
    edges=_edges(z)
    tabs=st.tabs(["Overview","Transactions","Network","Geography"])
    with tabs[0]:
        ev=lin[lin.CASE_ID.eq(cid)].copy(); ev["ts"]=pd.to_datetime(ev.ts)
        m1,m2,m3=st.columns(3)
        nodes=set(edges.Sender_account.astype(str))|set(edges.Receiver_account.astype(str))
        m1.metric("Counterparties",max(0,len(nodes)-1)); m2.metric("Cross-border",int(edges.cross_border.sum()))
        m3.metric("Countries",len(set(edges.sender_iso2.dropna())|set(edges.receiver_iso2.dropna())))
        st.subheader("Alert timeline"); st.scatter_chart(ev,x="ts",y="holdout_row_id",color="scenario")
    with tabs[1]:
        st.dataframe(edges.sort_values("ts"),width="stretch",hide_index=True)
    with tabs[2]:
        st.pyplot(_network(edges,r.account_id),width="stretch")
        st.caption("Directed graph of actual alerted transfers. Arrow = direction of funds; thicker edge = repeated transfers.")
    with tabs[3]:
        fig,g=_world_map(edges)
        st.plotly_chart(fig,width="stretch")
        st.caption("Country exposure for this investigation case. Hover over a country for sent/received alerted amounts and transaction volume.")
        cb=edges[edges.cross_border.eq(1)]
        if len(cb):
            routes=(cb.groupby(["sender_iso2","receiver_iso2"]).agg(transactions=("holdout_row_id","count"),amount=("Amount","sum"))
                    .reset_index().sort_values("amount",ascending=False))
            routes["route"]=routes.sender_iso2+" → "+routes.receiver_iso2
            st.subheader("Cross-border routes")
            st.dataframe(routes[["route","transactions","amount"]],width="stretch",hide_index=True)
    st.caption("The frozen ML score prioritizes human review; it is not an AML determination.")
