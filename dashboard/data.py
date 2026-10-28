from pathlib import Path
import pandas as pd
import streamlit as st
R=Path("results"); CM=R/"case_management_v3"
@st.cache_data
def load_dashboard():
    network_path=R/"dashboard/case_transaction_network_v3.csv"
    ml_score_path=R/"ml_case_v3/ml_case_scores.csv"
    ml_imp_path=R/"ml_case_v3/ml_feature_importance.csv"
    ml_det_path=R/"ml_detection/holdout_v1/ml_holdout_scores.csv"
    overlap_path=R/"comparison_rule_v3_vs_ml_v1/overlap_summary.csv"
    ml_dev_ops_path=R/"ml_detection/full_train_development/operating_points.csv"
    q=pd.read_csv(CM/"prioritized/case_priority_queue.csv")
    alerts=pd.read_csv(CM/"aggregated/alert.csv")
    alert_tx=pd.read_csv(CM/"aggregated/alert_transaction.csv")
    case_alert=pd.read_csv(CM/"aggregated_cases/case_alert.csv")
    metrics=pd.read_csv(R/"holdout_v3/final_metrics.csv")
    for x in ["case_created_at","last_alert_at"]:
        if x in q.columns:q[x]=pd.to_datetime(q[x],errors="coerce")
    for x in ["alert_created_at","last_trigger_at"]:
        if x in alerts.columns:alerts[x]=pd.to_datetime(alerts[x],errors="coerce")
    sc=alerts.groupby("scenario_id",as_index=False).agg(aggregated_alerts=("alert_id","count"),triggering_transactions=("transaction_count","sum"),unique_accounts=("primary_account_id","nunique"))
    return {"queue":q,"alerts":alerts,"alert_tx":alert_tx,"case_alert":case_alert,"metrics":metrics,"scenarios":sc,"network":pd.read_csv(network_path) if network_path.exists() else None,"ml_scores":pd.read_csv(ml_score_path) if ml_score_path.exists() else None,"ml_importance":pd.read_csv(ml_imp_path) if ml_imp_path.exists() else None,
            "ml_detection":pd.read_csv(ml_det_path) if ml_det_path.exists() else None,
            "overlap":pd.read_csv(overlap_path) if overlap_path.exists() else None,
            "ml_dev_ops":pd.read_csv(ml_dev_ops_path) if ml_dev_ops_path.exists() else None}
