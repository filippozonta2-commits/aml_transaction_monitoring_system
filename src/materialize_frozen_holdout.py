"""Materialize the exact frozen six-scenario pipeline on HOLDOUT.

DEVELOPMENT is context only. Scenario definitions, fixed-window anchors and
72-hour Deposit-Send logic are unchanged. This script produces HOLDOUT artifacts
but does NOT calculate or print AML performance metrics.
"""
from pathlib import Path
import argparse, glob, subprocess, sys
import numpy as np
import pandas as pd
from country_normalization import country_key

COLS=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
"Payment_currency","Received_currency","Sender_bank_location","Receiver_bank_location","Is_laundering","Laundering_type"]
FLAGS=["SCN_SMURFING","SCN_CASH_WITHDRAWAL","SCN_FAN_OUT","SCN_STRUCTURING","SCN_FAN_IN","SCN_DEPOSIT_SEND"]
ROLES={"SCN_SMURFING":"Sender_account","SCN_CASH_WITHDRAWAL":"Sender_account","SCN_FAN_OUT":"Sender_account",
"SCN_STRUCTURING":"Sender_account","SCN_FAN_IN":"Receiver_account","SCN_DEPOSIT_SEND":"Receiver_account"}

def load(path,period):
 d=pd.read_csv(path,usecols=COLS); d["ts"]=pd.to_datetime(pd.to_datetime(d.Date,errors="coerce").dt.strftime("%Y-%m-%d")+" "+d.Time.astype(str),errors="coerce")
 d=d.dropna(subset=["ts"]); d["Period"]=period; d["cross"]=(d.Sender_bank_location!=d.Receiver_bank_location).astype("int8"); d["fx"]=(d.Payment_currency!=d.Received_currency).astype("int8"); return d
def sw(data,start,days):
 x=data.copy(); x["w"]=np.floor((x.ts-start).dt.total_seconds()/86400/days).astype("int32")
 g=x.groupby(["w","Sender_account"]).agg(tx_count=("Amount","size"),aggregate_amount=("Amount","sum"),median_amount=("Amount","median"),unique_receivers=("Receiver_account","nunique"),cross_border_rate=("cross","mean"),currency_mismatch_rate=("fx","mean")).reset_index()
 return x.merge(g,on=["w","Sender_account"],how="left")
def rw(x,days=10):
 x=x.copy(); x["wr"]=x.ts.dt.floor(f"{days}D"); g=x.groupby(["wr","Receiver_account"]).agg(receiver_tx_count=("Amount","size"),receiver_aggregate_amount=("Amount","sum"),receiver_unique_senders=("Sender_account","nunique"),receiver_cross_border_rate=("cross","mean"),receiver_currency_mismatch_rate=("fx","mean")).reset_index(); return x.merge(g,on=["wr","Receiver_account"],how="left")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv")); p.add_argument("--holdout",type=Path,default=Path("data/temporal/SAML-D_holdout.csv")); p.add_argument("--out",type=Path,default=Path("results/holdout")); a=p.parse_args()
 dev=load(a.development,"development"); ho=load(a.holdout,"holdout"); start=ho.ts.min(); warm=dev[dev.ts.ge(start-pd.Timedelta(days=60))].copy(); warm["holdout_row_id"]=-1
 ho=ho.copy(); ho["holdout_row_id"]=np.arange(len(ho),dtype="int64"); data=pd.concat([warm,ho],ignore_index=True); f=pd.DataFrame({"holdout_row_id":ho.holdout_row_id})
 cw=sw(data[data.Payment_type.eq("Cash Withdrawal")],start,7); ids=set(cw.loc[(cw.tx_count.ge(5)&cw.aggregate_amount.ge(300)&cw.median_amount.le(300)&cw.Period.eq("holdout")),"holdout_row_id"]); f["SCN_CASH_WITHDRAWAL"]=f.holdout_row_id.isin(ids).astype("int8")
 sm=sw(data[data.Payment_type.eq("Cash Deposit")],start,45); ids=set(sm.loc[(sm.tx_count.ge(3)&sm.median_amount.lt(4000)&sm.aggregate_amount.ge(10000)&sm.cross_border_rate.le(.15)&sm.currency_mismatch_rate.le(.15)&sm.Period.eq("holdout")),"holdout_row_id"]); f["SCN_SMURFING"]=f.holdout_row_id.isin(ids).astype("int8")
 fo=sw(data,start,21); ids=set(fo.loc[(fo.unique_receivers.ge(3)&fo.tx_count.between(3,12)&((fo.cross_border_rate.ge(.20))|(fo.currency_mismatch_rate.ge(.20)))&fo.Period.eq("holdout")),"holdout_row_id"]); f["SCN_FAN_OUT"]=f.holdout_row_id.isin(ids).astype("int8")
 r=rw(ho); f["SCN_STRUCTURING"]=(r.Amount.lt(10000)&r.receiver_unique_senders.ge(5)&r.receiver_aggregate_amount.ge(20000)&((r.receiver_cross_border_rate.ge(.20))|(r.receiver_currency_mismatch_rate.ge(.25)))).astype("int8").to_numpy(); f["SCN_FAN_IN"]=(r.receiver_unique_senders.between(5,15)&r.receiver_tx_count.between(5,20)&((r.receiver_cross_border_rate.ge(.10))|(r.receiver_currency_mismatch_rate.ge(.20)))).astype("int8").to_numpy()
 a.out.mkdir(parents=True,exist_ok=True); base=ho[["holdout_row_id","ts","Sender_account","Receiver_account","Amount","Payment_type","Is_laundering","Laundering_type"]].merge(f,on="holdout_row_id")
 tmp=a.out/"deposit_send_flags"; subprocess.run([sys.executable,"src/materialize_deposit_send_spark.py","--train",str(a.development),"--development",str(a.holdout),"--output-dir",str(tmp)],check=True)
 parts=glob.glob(str(tmp/"part-*.csv")); ds=pd.concat([pd.read_csv(q) for q in parts]).rename(columns={"development_row_id":"holdout_row_id"}); out=base.merge(ds[["holdout_row_id","SCN_DEPOSIT_SEND"]],on="holdout_row_id",validate="one_to_one"); out["SCENARIO_COUNT"]=out[FLAGS].sum(axis=1); out["ANY_SCENARIO_ALERT"]=(out.SCENARIO_COUNT>0).astype("int8")
 x=out[out.ANY_SCENARIO_ALERT.eq(1)]; ps=[]
 for sc,ac in ROLES.items():
  z=x[x[sc].eq(1)][["holdout_row_id","ts","Amount","Is_laundering","Laundering_type",ac]].rename(columns={ac:"account_id"}); z["scenario"]=sc; ps.append(z)
 e=pd.concat(ps).dropna(subset=["account_id"]).sort_values(["account_id","ts","holdout_row_id"]); prev=e.groupby("account_id").ts.shift(); e["new"]=(prev.isna()|((e.ts-prev).dt.total_seconds()/3600>72)).astype(int); e["seq"]=e["new"].groupby(e.account_id).cumsum(); e["CASE_ID"]="HCASE-"+e.account_id.astype(str)+"-"+e.seq.astype(str).str.zfill(3)
 q=e.groupby(["CASE_ID","account_id"]).agg(case_start=("ts","min"),case_end=("ts","max"),scenario_events=("holdout_row_id","size"),transaction_alerts=("holdout_row_id","nunique"),total_alert_amount=("Amount","sum"),active_scenarios=("scenario","nunique")).reset_index(); q["duration_hours"]=(q.case_end-q.case_start).dt.total_seconds()/3600; sl=e.groupby("CASE_ID").scenario.agg(lambda s:" | ".join(sorted(set(s)))).rename("scenario_list"); q=q.merge(sl,on="CASE_ID"); q["multi_scenario_flag"]=(q.active_scenarios>1).astype("int8"); q["case_status"]="OPEN"; q["case_policy"]="72h_inactivity"
 ev=e.groupby("CASE_ID").agg(aml_positive=("Is_laundering",lambda s:int((s==1).any())),aml_event_count=("Is_laundering","sum")).reset_index(); types=e[e.Is_laundering.eq(1)].groupby("CASE_ID").Laundering_type.agg(lambda s:" | ".join(sorted(set(s.dropna().astype(str))))).rename("aml_typologies"); ev=ev.merge(types,on="CASE_ID",how="left")
 links=e[["CASE_ID","account_id","holdout_row_id","ts","scenario"]]; tx=pd.read_csv(a.holdout); tx["holdout_row_id"]=np.arange(len(tx)); z=links.merge(tx[["holdout_row_id","Sender_account","Receiver_account","Amount","Payment_type","Payment_currency","Received_currency","Sender_bank_location","Receiver_bank_location"]],on="holdout_row_id",validate="many_to_one"); z["cross_border"]=(z.Sender_bank_location!=z.Receiver_bank_location).astype(int); z["currency_mismatch"]=(z.Payment_currency!=z.Received_currency).astype(int); z["sender_iso2"]=z.Sender_bank_location.map(country_key); z["receiver_iso2"]=z.Receiver_bank_location.map(country_key); t=z.drop_duplicates(["CASE_ID","holdout_row_id"]).copy(); t["counterparty_id"]=np.where(t.Sender_account.astype(str).eq(t.account_id.astype(str)),t.Receiver_account,t.Sender_account)
 g=t.groupby("CASE_ID").agg(tx_amount_sum=("Amount","sum"),tx_amount_mean=("Amount","mean"),tx_amount_median=("Amount","median"),tx_amount_max=("Amount","max"),tx_amount_std=("Amount","std"),unique_counterparties=("counterparty_id","nunique"),payment_types=("Payment_type","nunique"),cross_border_rate=("cross_border","mean"),currency_mismatch_rate=("currency_mismatch","mean"),sender_countries=("sender_iso2","nunique"),receiver_countries=("receiver_iso2","nunique")).reset_index(); g.tx_amount_std=g.tx_amount_std.fillna(0); pt=pd.crosstab(t.CASE_ID,t.Payment_type,normalize="index").add_prefix("payment_share_").reset_index(); feat=q.merge(g,on="CASE_ID").merge(pt,on="CASE_ID",how="left").fillna(0); feat["alerts_per_hour"]=feat.transaction_alerts/(feat.duration_hours+1); feat["amount_per_alert"]=feat.tx_amount_sum/feat.transaction_alerts.clip(lower=1)
 out.to_csv(a.out/"unified_alert_table_holdout.csv",index=False); q.to_csv(a.out/"investigator_queue.csv",index=False); ev.to_csv(a.out/"case_evaluation_labels.csv",index=False); links.to_csv(a.out/"case_transaction_lineage.csv",index=False); feat.to_csv(a.out/"case_features_holdout.csv",index=False)
 print("\n=== HOLDOUT FROZEN PIPELINE MATERIALIZATION ==="); print(f"Transactions: {len(ho):,}"); print(f"Alerted transactions: {int(out.ANY_SCENARIO_ALERT.sum()):,}"); print(f"Cases: {len(q):,}"); print(f"Feature columns: {len(feat.columns)}"); print("No performance metrics were calculated or inspected."); print(f"Saved: {a.out}")
if __name__=="__main__": main()
