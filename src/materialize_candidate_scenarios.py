"""Materialize first-pass flags for the 12 new AML scenarios on DEVELOPMENT only.

Purpose: engineering/diagnostic pass, not threshold selection. No HOLDOUT access.
Graph-heavy second-order scenarios are implemented with bounded temporal joins
rather than unrestricted graph expansion.
"""
from pathlib import Path
import argparse, numpy as np, pandas as pd

FLAGS=["SCN_SINGLE_LARGE_TRANSACTION","SCN_UNUSUAL_AMOUNT","SCN_HIGH_TRANSACTION_VELOCITY",
"SCN_GATHER_SCATTER","SCN_SCATTER_GATHER","SCN_CIRCULAR_MOVEMENT","SCN_LAYERED_FAN_OUT",
"SCN_LAYERED_FAN_IN","SCN_HIGH_RISK_GEOGRAPHY","SCN_SANCTIONED_GEOGRAPHY",
"SCN_CROSS_BORDER_CURRENCY_MISMATCH","SCN_BEHAVIORAL_CHANGE"]

def main():
 p=argparse.ArgumentParser(); p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv")); p.add_argument("--out",type=Path,default=Path("results/candidate_scenarios")); a=p.parse_args()
 print("Loading DEVELOPMENT only...",flush=True)
 x=pd.read_csv(a.development); x["development_row_id"]=np.arange(len(x),dtype="int64")
 x["ts"]=pd.to_datetime(pd.to_datetime(x.Date,errors="coerce").dt.strftime("%Y-%m-%d")+" "+x.Time.astype(str),errors="coerce"); x=x.dropna(subset=["ts"]).copy()
 x["cross"]=(x.Sender_bank_location!=x.Receiver_bank_location); x["fx"]=(x.Payment_currency!=x.Received_currency)
 f=pd.DataFrame({"development_row_id":x.development_row_id})
 # Amount: global tail + sender-relative tail. Quantiles are candidate diagnostics, not frozen thresholds.
 q995=x.Amount.quantile(.995); f["SCN_SINGLE_LARGE_TRANSACTION"]=x.Amount.ge(q995).astype("int8").to_numpy()
 g=x.groupby("Sender_account").Amount.agg(["median","count"]).rename(columns={"median":"acct_med","count":"acct_n"}); z=x.join(g,on="Sender_account")
 ratio=x.Amount/(z.acct_med.replace(0,np.nan)); f["SCN_UNUSUAL_AMOUNT"]=((z.acct_n.ge(5))&ratio.ge(5)&x.Amount.ge(x.Amount.quantile(.90))).astype("int8").to_numpy()
 # Velocity / gather-scatter on 24h calendar buckets.
 x["day"]=x.ts.dt.floor("D")
 sg=x.groupby(["day","Sender_account"]).agg(out_n=("Amount","size"),out_receivers=("Receiver_account","nunique")).reset_index()
 rg=x.groupby(["day","Receiver_account"]).agg(in_n=("Amount","size"),in_senders=("Sender_account","nunique")).reset_index()
 z=x.merge(sg,on=["day","Sender_account"],how="left").merge(rg,left_on=["day","Sender_account"],right_on=["day","Receiver_account"],how="left",suffixes=("","_as_receiver"))
 f["SCN_HIGH_TRANSACTION_VELOCITY"]=(z.out_n.ge(max(5,int(sg.out_n.quantile(.99))))).astype("int8").to_numpy()
 f["SCN_GATHER_SCATTER"]=(z.in_senders.fillna(0).ge(3)&z.out_receivers.ge(3)).astype("int8").to_numpy()
 # Geography / FX. High-risk/sanctioned need governed reference data; do not infer them from labels.
 f["SCN_CROSS_BORDER_CURRENCY_MISMATCH"]=(x.cross&x.fx).astype("int8").to_numpy()
 f["SCN_HIGH_RISK_GEOGRAPHY"]=np.int8(0); f["SCN_SANCTIONED_GEOGRAPHY"]=np.int8(0)
 # Behavioral change: compare daily account behavior to earlier daily history, minimum 7 prior days.
 daily=x.groupby(["day","Sender_account"]).agg(n=("Amount","size"),amt=("Amount","sum")).reset_index().sort_values(["Sender_account","day"])
 daily["prior_n_mean"]=daily.groupby("Sender_account").n.transform(lambda s:s.shift().expanding(7).mean())
 daily["prior_amt_mean"]=daily.groupby("Sender_account").amt.transform(lambda s:s.shift().expanding(7).mean())
 daily["change"]=((daily.n>=3*daily.prior_n_mean)|(daily.amt>=4*daily.prior_amt_mean))&daily.prior_n_mean.notna()
 z=x.merge(daily[["day","Sender_account","change"]],on=["day","Sender_account"],how="left"); f["SCN_BEHAVIORAL_CHANGE"]=z.change.fillna(False).astype("int8").to_numpy()
 # Second-order graph candidates are scaffolded separately to avoid silently approximating path semantics.
 for col in ["SCN_SCATTER_GATHER","SCN_CIRCULAR_MOVEMENT","SCN_LAYERED_FAN_OUT","SCN_LAYERED_FAN_IN"]: f[col]=np.int8(0)
 a.out.mkdir(parents=True,exist_ok=True); f.to_csv(a.out/"candidate_flags_development.csv",index=False)
 summary=pd.DataFrame({"scenario":FLAGS,"triggered":[int(f[k].sum()) for k in FLAGS]}); summary.to_csv(a.out/"candidate_summary.csv",index=False)
 print("\n=== 12-SCENARIO DEVELOPMENT CANDIDATE PASS ==="); print(summary.to_string(index=False))
 print("\nZero-count geography flags await governed country-risk/sanctions mapping.")
 print("Zero-count second-order graph flags await exact bounded-path implementation.")
 print("No thresholds frozen. No HOLDOUT accessed."); print(f"Saved: {a.out}",flush=True)
if __name__=="__main__": main()
