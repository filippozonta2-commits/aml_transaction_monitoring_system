"""Build HOLDOUT case features for the frozen DEVELOPMENT-selected ML prioritizer.

Uses the current V3/V1 operational cases and the V3/V1 case-context network mart.
No HOLDOUT AML labels are read. Feature definitions mirror the DEVELOPMENT model
where possible; missing model columns are filled later by the scoring contract.
"""
from pathlib import Path
import argparse, numpy as np, pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--queue",type=Path,default=Path("results/case_management_v3/prioritized/case_priority_queue.csv"))
    p.add_argument("--network",type=Path,default=Path("results/dashboard/case_transaction_network_v3.csv"))
    p.add_argument("--out",type=Path,default=Path("results/ml_case_v3/case_features_holdout.csv"))
    a=p.parse_args()
    q=pd.read_csv(a.queue); z=pd.read_csv(a.network)
    z["case_id"]=z.case_id.astype(str)
    z["subject_id"]=z.subject_id.astype(str)
    z["Sender_account"]=z.Sender_account.astype(str); z["Receiver_account"]=z.Receiver_account.astype(str)
    z["counterparty_id"]=np.where(z.Sender_account.eq(z.subject_id),z.Receiver_account,z.Sender_account)
    if "cross_border" not in z: z["cross_border"]=(z.sender_iso2!=z.receiver_iso2).astype("int8")
    if "currency_mismatch" not in z: z["currency_mismatch"]=(z.Payment_currency!=z.Received_currency).astype("int8")
    g=z.groupby("case_id").agg(
      tx_amount_sum=("Amount","sum"),tx_amount_mean=("Amount","mean"),tx_amount_median=("Amount","median"),
      tx_amount_max=("Amount","max"),tx_amount_std=("Amount","std"),unique_counterparties=("counterparty_id","nunique"),
      payment_types=("Payment_type","nunique"),cross_border_rate=("cross_border","mean"),
      currency_mismatch_rate=("currency_mismatch","mean"),sender_countries=("sender_iso2","nunique"),
      receiver_countries=("receiver_iso2","nunique")).reset_index()
    g["tx_amount_std"]=g.tx_amount_std.fillna(0.0)
    pt=pd.crosstab(z.case_id,z.Payment_type,normalize="index").add_prefix("payment_share_").reset_index()
    x=q.copy(); x["case_id"]=x.case_id.astype(str)
    x=x.merge(g,on="case_id",how="left",validate="one_to_one").merge(pt,on="case_id",how="left",validate="one_to_one")
    x["CASE_ID"]=x.case_id; x["account_id"]=x.subject_id
    x["case_start"]=x.case_created_at; x["case_end"]=x.last_alert_at
    x["transaction_alerts"]=x.transaction_count
    x["active_scenarios"]=x.scenario_count
    dur=(pd.to_datetime(x.case_end)-pd.to_datetime(x.case_start)).dt.total_seconds().div(3600).clip(lower=0)
    x["duration_hours"]=dur
    x["alerts_per_hour"]=x.transaction_alerts/(dur+1.0)
    x["amount_per_alert"]=x.tx_amount_sum/x.transaction_alerts.clip(lower=1)
    x["multi_scenario_flag"]=x.scenario_count.gt(1).astype("int8")
    x["scenario_events"]=x.alert_count
    x["total_alert_amount"]=x.tx_amount_sum
    a.out.parent.mkdir(parents=True,exist_ok=True); x.to_csv(a.out,index=False)
    print("\n=== HOLDOUT ML CASE FEATURES ===")
    print(f"Cases: {len(x):,} | columns: {len(x.columns)} | network rows: {len(z):,}")
    print("No HOLDOUT AML labels read. Detection and V3/V1 case construction unchanged.")
    print(f"Saved: {a.out}")
if __name__=="__main__":main()
