import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import pycountry

def _network(edges,account):
    G=nx.DiGraph()
    if "network_eligible" in edges.columns: edges=edges[edges.network_eligible.eq(1)]
    for _,x in edges.iterrows():
        u=str(x["Sender_account"]); v=str(x["Receiver_account"])
        if G.has_edge(u,v): G[u][v]["count"]+=1
        else:G.add_edge(u,v,count=1)
    fig,ax=plt.subplots(figsize=(11,6))
    pos=nx.spring_layout(G,seed=42,k=max(.5,2/(max(len(G),1)**.5)))
    nx.draw_networkx_nodes(G,pos,node_size=[1500 if n==str(account) else 430 for n in G],ax=ax)
    nx.draw_networkx_edges(G,pos,width=[.8+min(4,G[u][v]["count"]*.35) for u,v in G.edges],
        arrows=True,arrowstyle="-|>",arrowsize=22,connectionstyle="arc3,rad=0.04",
        min_source_margin=18,min_target_margin=22,alpha=.60,ax=ax)
    nx.draw_networkx_labels(G,pos,font_size=7,ax=ax); ax.axis("off"); return fig

def _iso3(x):
    try:return pycountry.countries.get(alpha_2=str(x)).alpha_3
    except:return None

def _world_map(z):
    s=z.groupby("sender_iso2").agg(sent=("Amount","sum"),outgoing=("Amount","size"))
    r=z.groupby("receiver_iso2").agg(received=("Amount","sum"),incoming=("Amount","size"))
    g=s.join(r,how="outer").fillna(0).reset_index().rename(columns={"index":"iso2","sender_iso2":"iso2","receiver_iso2":"iso2"})
    g["iso3"]=g.iso2.map(_iso3); g["total_amount"]=g.sent+g.received; g["transactions"]=g.outgoing+g.incoming
    fig=px.choropleth(g,locations="iso3",color="total_amount",hover_name="iso2",
        hover_data={"iso3":False,"sent":":,.0f","received":":,.0f","transactions":":,.0f"},
        projection="natural earth",labels={"total_amount":"Case amount"})
    fig.update_geos(showcoastlines=True,showland=True,fitbounds="locations")
    fig.update_layout(height=500,margin=dict(l=0,r=0,t=10,b=0),coloraxis_colorbar_title="Amount")
    return fig

def render(d):
    q=d["queue"].copy()
    alerts=d["alerts"]
    ca=d["case_alert"]
    at=d["alert_tx"]
    net=d.get("network")
    strategy=d.get("strategy","Scenario-based")
    ml=d.get("ml_scores")
    if ml is not None:
        q=q.merge(ml[["case_id","ml_probability","ml_score","ml_rank"]],on="case_id",how="left",validate="one_to_one")
    st.header("Case Investigation")
    order="ml_rank" if strategy=="Machine Learning" and "ml_rank" in q.columns else "queue_rank"
    cid=st.selectbox("Select investigation case",q.sort_values(order).case_id.tolist())
    r=q[q.case_id.eq(cid)].iloc[0]
    st.markdown(f"### {cid}  ·  Subject {r.subject_id}")
    a,b,c,e,f=st.columns(5)
    a.metric("Queue rank",f"#{int(r.ml_rank):,}" if strategy=="Machine Learning" and pd.notna(r.get("ml_rank")) else f"#{int(r.queue_rank):,}"); b.metric("Priority",r.queue_priority)
    c.metric("Risk score",f"{int(r.risk_score)}"); e.metric("Alerts",f"{int(r.alert_count):,}")
    f.metric("Transactions",f"{int(r.transaction_count):,}")
    if "ml_rank" in q.columns and pd.notna(r.get("ml_rank")):
        x1,x2,x3=st.columns(3); x1.metric("Scenario rank",f"#{int(r.queue_rank):,}"); x2.metric("ML rank",f"#{int(r.ml_rank):,}",delta=f"{int(r.queue_rank-r.ml_rank):+,} positions"); x3.metric("ML score",f"{float(r.ml_score):.1f}")
    st.caption(f"Scenarios: {r.scenarios}")
    st.caption(f"Policy-linked: {bool(r.policy_flag)} · {r.priority_reason}")

    ids=ca.loc[ca.case_id.eq(cid),"alert_id"]
    ad=alerts[alerts.alert_id.isin(ids)].copy()
    tx=at[at.alert_id.isin(ids)].copy()

    # Resolve case network/context before rendering any tab because
    # Transaction Evidence also uses ISO2 fields from this mart.
    z=None
    if net is not None:
        case_col="case_id" if "case_id" in net.columns else ("CASE_ID" if "CASE_ID" in net.columns else None)
        if case_col:
            z=net[net[case_col].astype(str).eq(str(cid))].copy()

    tabs=st.tabs(["Overview","Alerts","Transaction Evidence","Network","Geography"])
    with tabs[0]:
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Scenario count",int(r.scenario_count)); m2.metric("Aggregated alerts",len(ad))
        m3.metric("Evidence links",len(tx)); m4.metric("Policy flag","YES" if bool(r.policy_flag) else "NO")
        if len(ad):
            timeline=ad.copy(); timeline["alert_created_at"]=pd.to_datetime(timeline["alert_created_at"],errors="coerce")
            chart=timeline.groupby([timeline.alert_created_at.dt.date,"scenario_id"]).size().reset_index(name="alerts")
            chart["date"]=pd.to_datetime(chart["alert_created_at"])
            st.subheader("Alert timeline"); st.line_chart(chart,x="date",y="alerts",color="scenario_id")
        st.subheader("Why this case is prioritized")
        st.write(r.priority_reason)
    with tabs[1]:
        cols=[x for x in ["alert_id","scenario_id","primary_account_id","alert_created_at","last_trigger_at","transaction_count","alert_amount","policy_flag","trigger_reason"] if x in ad.columns]
        st.dataframe(ad[cols].sort_values("alert_created_at"),width="stretch",hide_index=True)
    with tabs[2]:
        evidence=tx.copy()
        if z is not None and len(z) and "transaction_id" in z.columns:
            geo_cols=[x for x in ["transaction_id","sender_iso2","receiver_iso2","Sender_bank_location","Receiver_bank_location"] if x in z.columns]
            evidence=evidence.merge(z[geo_cols].drop_duplicates("transaction_id"),on="transaction_id",how="left")
        preferred=[x for x in ["alert_id","transaction_id","scenario_id","transaction_timestamp","sender_iso2","receiver_iso2","Sender_bank_location","Receiver_bank_location","contribution_role","trigger_value","linked_at"] if x in evidence.columns]
        rest=[x for x in evidence.columns if x not in preferred]
        evidence=evidence[preferred+rest]
        st.dataframe(evidence.sort_values(["alert_id","linked_at"]) if "linked_at" in evidence.columns else evidence,width="stretch",hide_index=True)

    with tabs[3]:
        if z is None or not len(z):
            st.info("The current network mart is legacy or has no rows for this V1 case. Rebuild it against the V3/V1 case lineage before using this view.")
        elif {"Sender_account","Receiver_account"}.issubset(z.columns):
            st.pyplot(_network(z,r.subject_id),width="stretch")
            st.caption("Sender → Receiver account network. Cash flows are excluded when network_eligible=0.")
    with tabs[4]:
        if z is None or not len(z) or not {"sender_iso2","receiver_iso2","Amount"}.issubset(z.columns):
            st.info("Geography requires the V3/V1 case network mart.")
        else:
            st.plotly_chart(_world_map(z),width="stretch")
            if "cross_border" in z.columns:
                cb=z[z.cross_border.eq(1)]
                if len(cb):
                    routes=cb.groupby(["sender_iso2","receiver_iso2"]).agg(transactions=("Amount","size"),amount=("Amount","sum")).reset_index().sort_values("amount",ascending=False)
                    routes["route"]=routes.sender_iso2+" → "+routes.receiver_iso2
                    st.subheader("Cross-border routes"); st.dataframe(routes[["sender_iso2","receiver_iso2","route","transactions","amount"]],width="stretch",hide_index=True)
            st.subheader("Geographic transaction detail")
            geo_detail=z[[x for x in ["transaction_id","sender_iso2","receiver_iso2","Sender_bank_location","Receiver_bank_location","Amount","scenario","is_alerted_transaction"] if x in z.columns]].copy()
            st.dataframe(geo_detail,width="stretch",hide_index=True)
    st.caption("Operational priority supports investigator workflow; it is not an AML determination.")
