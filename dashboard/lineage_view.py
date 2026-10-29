import pandas as pd
import streamlit as st

def render(d,mode="lineage"):
    if mode=="ml":
        st.header("ML Detection Analytics")
        st.caption("Frozen transaction-level XGBoost Detection V1 · independent alternative to Rule-Based V3")
        ml=d.get("ml_detection"); ops=d.get("ml_dev_ops"); ov=d.get("overlap")
        if ml is None:
            st.warning("Frozen ML detector output is unavailable.")
            return
        n=len(ml); alerts=int(ml.ML_ALERT_V1.sum()); positives=int(ml.Is_laundering.sum())
        hits=int(ml.loc[ml.ML_ALERT_V1.eq(1),"Is_laundering"].sum())
        a,b,c1,e=st.columns(4)
        a.metric("Threshold","0.653366"); b.metric("HOLDOUT alerts",f"{alerts:,}")
        c1.metric("Precision",f"{hits/alerts:.2%}"); e.metric("Recall",f"{hits/positives:.2%}")
        left,right=st.columns(2)
        with left:
            st.subheader("Probability distribution")
            z=ml.copy(); z["score_band"]=(z.ml_probability*10).astype(int).clip(0,9)
            dist=z.score_band.value_counts().sort_index()
            dist.index=[f"{i/10:.1f}–{(i+1)/10:.1f}" for i in dist.index]
            st.bar_chart(dist)
        with right:
            st.subheader("DEVELOPMENT operating points")
            if ops is not None:
                show=ops[["alert_rate","alerts","precision","recall","threshold"]].copy()
                st.dataframe(show,width="stretch",hide_index=True)
        if ov is not None:
            st.subheader("Frozen ML V1 vs Rule V3 AML overlap")
            st.bar_chart(ov.set_index("group")["aml_positives"])
        st.subheader("Model governance")
        st.markdown("**Model:** XGBoost · **Training:** full TRAIN · **model/threshold selection:** DEVELOPMENT · **threshold:** 0.653366 · **HOLDOUT:** evaluation only.")
        st.info("ML V1 is a transaction-level detection engine. It is not the older case-prioritization model.")
        return
    st.header("Governance & Lineage")
    st.caption("Two independent frozen detectors, each with its own downstream operational workflow.")
    strategy = d.get("strategy", "Scenario-based")
    is_ml = strategy == "Machine Learning"
    st.subheader("Machine Learning · XGBoost Detection V1" if is_ml else "Rule-Based · Frozen V3 (11 scenarios)")
    if is_ml:
        scores = d.get("ml_detection")
        agg = d.get("ml_alerts")
        cases = d.get("ml_cases")
        bridge = d.get("ml_alert_tx")
        links = d.get("ml_case_alert")
        if scores is not None:
            tx_alerts = int(scores.ML_ALERT_V1.sum())
            st.metric("Frozen transaction alerts", f"{tx_alerts:,}")
        if agg is not None:
            st.metric("Aggregated ML alerts", f"{len(agg):,}")
        if cases is not None:
            st.metric("Independent ML cases", f"{len(cases):,}")
        if bridge is not None and agg is not None:
            missing = set(bridge.alert_id) - set(agg.alert_id)
            if missing:
                st.error(f"{len(missing):,} ML alert-to-transaction links lack an alert header.")
        if links is not None and agg is not None:
            missing = set(links.alert_id) - set(agg.alert_id)
            if missing:
                st.error(f"{len(missing):,} ML case-to-alert links lack an alert header.")
        st.graphviz_chart("""digraph {
          rankdir=TB;
          node [shape=box, style="rounded,filled", fillcolor="#111827", fontcolor="white", color="#475569"];
          Raw[label="HOLDOUT transactions"];
          Feat[label="Frozen ML feature engineering"];
          Model[label="XGBoost Detection V1"];
          Threshold[label="Frozen decision threshold"];
          Tx[label="ML transaction alerts"];
          Agg[label="Account + 7-day inactivity session\\nAggregated ML alerts"];
          Cases[label="Account + 30-day inactivity session\\nIndependent ML cases"];
          Queue[label="ML case priority queue\\nMax frozen transaction probability"];
          Review[label="Human investigation"];
          Raw->Feat->Model->Threshold->Tx->Agg->Cases->Queue->Review;
        }""")
        st.caption("The operational account is the originating sender. Account-level grouping is not model retraining or label-based selection.")
        if agg is None or cases is None:
            st.info("ML case artifacts are not available yet. Run the ML alert aggregation and case generation scripts.")
    else:
        alerts = d.get("alerts")
        cases = d.get("queue")
        bridge = d.get("alert_tx")
        links = d.get("case_alert")
        if alerts is not None:
            st.metric("Aggregated rule-based alerts", f"{len(alerts):,}")
        if cases is not None:
            st.metric("Rule-based investigation cases", f"{len(cases):,}")
        if bridge is not None and alerts is not None:
            missing = set(bridge.alert_id) - set(alerts.alert_id)
            if missing:
                st.error(f"{len(missing):,} Rule alert-to-transaction links lack an alert header.")
        if links is not None and alerts is not None:
            missing = set(links.alert_id) - set(alerts.alert_id)
            if missing:
                st.error(f"{len(missing):,} Rule case-to-alert links lack an alert header.")
        st.graphviz_chart("""digraph {
          rankdir=TB;
          node [shape=box, style="rounded,filled", fillcolor="#111827", fontcolor="white", color="#475569"];
          Raw[label="HOLDOUT transactions"];
          Rules[label="Frozen Rule-Based V3\\n11 scenarios"];
          Tx[label="Transaction-level scenario triggers"];
          Agg[label="Account + scenario + inactivity session\\nAggregated rule alerts"];
          Cases[label="Account + 30-day inactivity session\\nRule-based cases"];
          Queue[label="Transparent operational priority"];
          Review[label="Human investigation"];
          Raw->Rules->Tx->Agg->Cases->Queue->Review;
        }""")
        st.caption("Scenario-specific aggregation windows are operational workflow choices, not changes to frozen rule thresholds.")
    st.subheader("Controls and evaluation boundaries")
    st.markdown("""
    - **Detection:** TRAIN/DEVELOPMENT are used for model development; frozen detectors are evaluated on chronological HOLDOUT.
    - **Operationalization:** Alert aggregation, case sessionization and queue ranking happen after detection and do not use AML ground-truth labels.
    - **Traceability:** Case → aggregated alert → original triggering transaction is preserved in link tables.
    - **Distinct populations:** ML cases and rule-based cases are not interchangeable. Case counts are not transaction-level recall.
    - **Human decision:** Neither alert nor case priority constitutes an AML finding or an automatic SAR decision.
    """)
