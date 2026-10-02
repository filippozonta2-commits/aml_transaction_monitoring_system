from pathlib import Path
import pandas as pd
import streamlit as st

R=Path("results")

@st.cache_data
def load_dashboard():
    network_path=R/"dashboard/case_transaction_network.csv"
    return {
        "queue":pd.read_csv(R/"dashboard/ranked_investigator_queue.csv"),
        "kpis":pd.read_csv(R/"dashboard/dashboard_kpis.csv").iloc[0],
        "scenarios":pd.read_csv(R/"dashboard/scenario_summary.csv"),
        "lineage":pd.read_csv(R/"holdout/case_transaction_lineage.csv"),
        "alerts":pd.read_csv(R/"holdout/unified_alert_table_holdout.csv"),
        "network":pd.read_csv(network_path) if network_path.exists() else None,
    }
