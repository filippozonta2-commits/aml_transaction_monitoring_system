"""Materialize the frozen V3 non-Deposit-Send controls on HOLDOUT exactly once.

This is validation only. No threshold selection or tuning is performed here.
DEVELOPMENT is used only where a frozen rule requires prior-history context or
a DEVELOPMENT-frozen numeric threshold.
"""
from pathlib import Path
import argparse, numpy as np, pandas as pd
from materialize_frozen_scenarios import load, sender_windows, receiver_windows
from transaction_semantics import add_transaction_semantics

def unusual_z4_with_history(history, holdout):
 h=history.copy(); q=holdout.copy()
 h["_period"]="history"; q["_period"]="holdout"; q["_holdout_row_id"]=np.arange(len(q),dtype="int64")
 h["_holdout_row_id"]=-1
 x=pd.concat([h,q],ignore_index=True)
 x["_ts"]=pd.to_datetime(x["Date"].astype(str)+" "+x["Time"].astype(str),errors="coerce")
 x["_ord"]=np.arange(len(x)); x=x.sort_values(["Sender_account","_ts","_ord"])
 grp=x.groupby("Sender_account",sort=False)["Amount"]
 x["_hist_n"]=grp.cumcount()
 x["_hist_avg"]=grp.transform(lambda v:v.shift().expanding().mean())
 x["_hist_sd"]=grp.transform(lambda v:v.shift().expanding().std())
 x["_z"]=(x["Amount"]-x["_hist_avg"])/x["_hist_sd"]
 z=x[x["_period"].eq("holdout")].sort_values("_holdout_row_id")
 return (z["_hist_n"].ge(10)&z["_hist_sd"].gt(0)&z["_z"].ge(4)).astype("int8").to_numpy()

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
 p.add_argument("--holdout",type=Path,default=Path("data/temporal/SAML-D_holdout.csv"))
 p.add_argument("--out",type=Path,default=Path("results/holdout_v3/non_deposit_send.csv"))
 a=p.parse_args()

 raw_dev=pd.read_csv(a.development); raw_hold=pd.read_csv(a.holdout)
 raw_dev["Amount"]=pd.to_numeric(raw_dev["Amount"],errors="coerce")
 raw_hold["Amount"]=pd.to_numeric(raw_hold["Amount"],errors="coerce")
 # Frozen DEVELOPMENT q99.5 threshold; never estimated from HOLDOUT.
 large_floor=float(raw_dev["Amount"].quantile(.995))

 dev=load(a.development,"development"); hold=load(a.holdout,"holdout")
 hold=hold.copy(); hold["holdout_row_id"]=np.arange(len(hold),dtype="int64")
 start=hold["ts"].min()
 warm=dev[dev["ts"].ge(start-pd.Timedelta(days=60))].copy(); warm["holdout_row_id"]=-1
 data=pd.concat([warm,hold],ignore_index=True)
 f=pd.DataFrame({"holdout_row_id":hold["holdout_row_id"]})

 cw=sender_windows(data[data["Payment_type"].eq("Cash Withdrawal")].copy(),start,7)
 m=cw["tx_count"].ge(5)&cw["aggregate_amount"].ge(300)&cw["median_amount"].le(300)
 ids=set(cw.loc[m&cw["Period"].eq("holdout"),"holdout_row_id"].astype(int))
 f["SCN_CASH_WITHDRAWAL"]=f.holdout_row_id.isin(ids).astype("int8")

 sm=sender_windows(data[data["Payment_type"].eq("Cash Deposit")].copy(),start,45)
 m=(sm["tx_count"].ge(3)&sm["median_amount"].lt(4000)&sm["aggregate_amount"].ge(10000)&
    sm["cross_border_rate"].le(.15)&sm["currency_mismatch_rate"].le(.15))
 ids=set(sm.loc[m&sm["Period"].eq("holdout"),"holdout_row_id"].astype(int))
 f["SCN_SMURFING"]=f.holdout_row_id.isin(ids).astype("int8")

 fo=sender_windows(data,start,21)
 m=(fo["unique_receivers"].ge(3)&fo["tx_count"].between(3,12)&
    (fo["cross_border_rate"].ge(.20)|fo["currency_mismatch_rate"].ge(.20)))
 ids=set(fo.loc[m&fo["Period"].eq("holdout"),"holdout_row_id"].astype(int))
 f["SCN_FAN_OUT"]=f.holdout_row_id.isin(ids).astype("int8")

 r=receiver_windows(hold,10)
 f["SCN_STRUCTURING"]=(r.Amount.lt(10000)&r.receiver_unique_senders.ge(5)&
   r.receiver_aggregate_amount.ge(20000)&
   (r.receiver_cross_border_rate.ge(.20)|r.receiver_currency_mismatch_rate.ge(.25))).astype("int8").to_numpy()
 f["SCN_FAN_IN"]=(r.receiver_unique_senders.between(5,15)&r.receiver_tx_count.between(5,20)&
   (r.receiver_cross_border_rate.ge(.10)|r.receiver_currency_mismatch_rate.ge(.20))).astype("int8").to_numpy()

 f["SCN_UNUSUAL_AMOUNT_Z4"]=unusual_z4_with_history(raw_dev,raw_hold)
 f["SCN_SINGLE_LARGE_TRANSACTION"]=raw_hold["Amount"].ge(large_floor).astype("int8").to_numpy()

 sem=add_transaction_semantics(raw_hold)
 sem["ts"]=pd.to_datetime(pd.to_datetime(sem.Date,errors="coerce").dt.strftime("%Y-%m-%d")+" "+sem.Time.astype(str),errors="coerce")
 sem["day"]=sem.ts.dt.floor("D")
 sg=sem.groupby(["day","Sender_account"]).agg(out_receivers=("Receiver_account","nunique")).reset_index()
 rg=sem.groupby(["day","Receiver_account"]).agg(in_senders=("Sender_account","nunique")).reset_index()
 z=sem.merge(sg,on=["day","Sender_account"],how="left").merge(
   rg,left_on=["day","Sender_account"],right_on=["day","Receiver_account"],how="left",suffixes=("","_as_receiver"))
 f["SCN_GATHER_SCATTER"]=(z.in_senders.fillna(0).ge(3)&z.out_receivers.ge(3)).astype("int8").to_numpy()

 meta=raw_hold[["Is_laundering","Laundering_type"]].copy()
 meta["holdout_row_id"]=np.arange(len(meta),dtype="int64")
 out=meta.merge(f,on="holdout_row_id",validate="one_to_one")
 a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
 print("=== FROZEN V3 NON-DEPOSIT-SEND — HOLDOUT ===")
 print(f"Rows: {len(out):,} | DEVELOPMENT-frozen large-transaction floor: {large_floor:,.2f}")
 for c in [x for x in f.columns if x.startswith("SCN_")]: print(f"{c}: {int(f[c].sum()):,}")
 print("HOLDOUT labels were not used to construct any flag.")
 print("Saved:",a.out)

if __name__=="__main__": main()
