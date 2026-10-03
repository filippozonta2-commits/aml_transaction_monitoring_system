"""Build V3/V1 case-context network mart for the investigation dashboard."""
from pathlib import Path
import argparse, numpy as np, pandas as pd
from country_normalization import country_key
from transaction_semantics import add_transaction_semantics
TX=["Date","Time","Sender_account","Receiver_account","Amount","Payment_type","Payment_currency","Received_currency","Sender_bank_location","Receiver_bank_location"]

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--holdout",type=Path,default=Path("data/temporal/SAML-D_holdout.csv"))
 p.add_argument("--queue",type=Path,default=Path("results/case_management_v3/prioritized/case_priority_queue.csv"))
 p.add_argument("--case-alert",type=Path,default=Path("results/case_management_v3/aggregated_cases/case_alert.csv"))
 p.add_argument("--alert-tx",type=Path,default=Path("results/case_management_v3/aggregated/alert_transaction.csv"))
 p.add_argument("--alerts",type=Path,default=Path("results/case_management_v3/aggregated/alert.csv"))
 p.add_argument("--out",type=Path,default=Path("results/dashboard/case_transaction_network_v3.csv"))
 p.add_argument("--summary",type=Path,default=Path("results/dashboard/network_complexity_v3.csv"))
 p.add_argument("--context-hours",type=int,default=72)
 a=p.parse_args()
 print("Loading V3/V1 inputs...",flush=True)
 q=pd.read_csv(a.queue); ca=pd.read_csv(a.case_alert); at=pd.read_csv(a.alert_tx); al=pd.read_csv(a.alerts)
 raw=add_transaction_semantics(pd.read_csv(a.holdout,usecols=TX))
 raw["holdout_row_id"]=np.arange(len(raw)); raw["transaction_id"]=raw.holdout_row_id.map(lambda i:f"TXN-HO-{int(i):010d}")
 raw["ts"]=pd.to_datetime(raw.Date.astype(str)+" "+raw.Time.astype(str),errors="coerce")
 q["case_created_at"]=pd.to_datetime(q.case_created_at); q["last_alert_at"]=pd.to_datetime(q.last_alert_at)
 links=ca[["case_id","alert_id"]].merge(at[["alert_id","transaction_id","scenario_id"]],on="alert_id",how="left")
 scen=links.groupby(["case_id","transaction_id"])["scenario_id"].agg(lambda s:" | ".join(sorted(set(s.dropna().astype(str))))).to_dict()
 sender={str(k):g.index.to_numpy() for k,g in raw.groupby(raw.Sender_account.astype(str),sort=False)}
 receiver={str(k):g.index.to_numpy() for k,g in raw.groupby(raw.Receiver_account.astype(str),sort=False)}
 chunks=[]; stats=[]; delta=pd.Timedelta(hours=a.context_hours)
 for i,x in enumerate(q.itertuples(index=False),1):
  acct=str(x.subject_id); ids=np.union1d(sender.get(acct,[]),receiver.get(acct,[])); z=raw.loc[ids]
  z=z[z.ts.between(x.case_created_at-delta,x.last_alert_at+delta)].copy()
  z["case_id"]=x.case_id; z["subject_id"]=acct
  z["scenario"]=[scen.get((x.case_id,t),"CONTEXT") for t in z.transaction_id]
  z["is_alerted_transaction"]=(z.scenario!="CONTEXT").astype("int8")
  z["sender_iso2"]=z.Sender_bank_location.map(country_key); z["receiver_iso2"]=z.Receiver_bank_location.map(country_key)
  z["cross_border"]=(z.geo_pair_applicable.eq(1)&(z.sender_iso2!=z.receiver_iso2)).astype("int8")
  z["currency_mismatch"]=(z.currency_pair_applicable.eq(1)&(z.Payment_currency!=z.Received_currency)).astype("int8")
  chunks.append(z)
  stats.append((x.case_id,acct,len(z),z.loc[z.network_eligible.eq(1),"counterparty_account"].nunique(),int(z.cross_border.sum()),int(z.is_alerted_transaction.sum())))
  if i%1000==0:print(f"Cases processed: {i:,}/{len(q):,}",flush=True)
 out=pd.concat(chunks,ignore_index=True)
 sm=pd.DataFrame(stats,columns=["case_id","subject_id","network_transactions","unique_counterparties","cross_border_transactions","alerted_transactions"]).sort_values(["unique_counterparties","network_transactions"],ascending=False)
 a.out.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.out,index=False);sm.to_csv(a.summary,index=False)
 print("\n=== V3/V1 CASE NETWORK MART ===");print(f"Network rows: {len(out):,} | Cases: {out.case_id.nunique():,}")
 print(sm.head(15).to_string(index=False));print(f"\nSaved: {a.out}\nSaved: {a.summary}")
if __name__=="__main__":main()
