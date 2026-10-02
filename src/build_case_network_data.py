"""Build full case-context networks efficiently with indexed account slices."""
from pathlib import Path
import argparse, numpy as np, pandas as pd
from country_normalization import country_key
from transaction_semantics import add_transaction_semantics
TX=["Date","Time","Sender_account","Receiver_account","Amount","Payment_type","Payment_currency","Received_currency","Sender_bank_location","Receiver_bank_location"]

def main():
 p=argparse.ArgumentParser(); p.add_argument("--holdout",type=Path,default=Path("data/temporal/SAML-D_holdout.csv")); p.add_argument("--lineage",type=Path,default=Path("results/holdout/case_transaction_lineage.csv")); p.add_argument("--queue",type=Path,default=Path("results/dashboard/ranked_investigator_queue.csv")); p.add_argument("--out",type=Path,default=Path("results/dashboard/case_transaction_network.csv")); p.add_argument("--summary",type=Path,default=Path("results/dashboard/network_complexity.csv")); a=p.parse_args()
 print("Loading inputs...",flush=True)
 links=pd.read_csv(a.lineage); q=pd.read_csv(a.queue); tx=add_transaction_semantics(pd.read_csv(a.holdout,usecols=TX))
 tx["holdout_row_id"]=np.arange(len(tx)); tx["ts"]=pd.to_datetime(tx.Date.astype(str)+" "+tx.Time.astype(str),errors="coerce")
 q["case_start"]=pd.to_datetime(q.case_start); q["case_end"]=pd.to_datetime(q.case_end)
 print(f"Transactions: {len(tx):,} | Cases: {len(q):,}",flush=True)
 scen=links.groupby("holdout_row_id")["scenario"].agg(lambda s:" | ".join(sorted(set(s.astype(str))))).to_dict()
 sender={str(k):g.index.to_numpy() for k,g in tx.groupby(tx.Sender_account.astype(str),sort=False)}
 receiver={str(k):g.index.to_numpy() for k,g in tx.groupby(tx.Receiver_account.astype(str),sort=False)}
 print("Account index ready. Building case windows...",flush=True)
 chunks=[]; stats=[]; delta=pd.Timedelta(hours=72)
 for i,x in enumerate(q.itertuples(index=False),1):
  acct=str(x.account_id); ids=np.union1d(sender.get(acct,[]),receiver.get(acct,[]))
  z=tx.loc[ids]
  z=z[z.ts.between(x.case_start-delta,x.case_end+delta)].copy()
  z["CASE_ID"]=x.CASE_ID; z["account_id"]=x.account_id
  z["scenario"]=z.holdout_row_id.map(scen).fillna("CONTEXT"); z["is_alerted_transaction"]=(z.scenario!="CONTEXT").astype("int8")
  z["sender_iso2"]=z.Sender_bank_location.map(country_key); z["receiver_iso2"]=z.Receiver_bank_location.map(country_key)
  z["cross_border"]=(z.sender_iso2!=z.receiver_iso2).astype("int8"); z["currency_mismatch"]=(z.Payment_currency!=z.Received_currency).astype("int8")
  z["counterparty_id"]=np.where(z.Sender_account.astype(str).eq(acct),z.Receiver_account,z.Sender_account)
  chunks.append(z)
  stats.append((x.CASE_ID,x.account_id,len(z),z.loc[z.network_eligible.eq(1),"counterparty_account"].nunique(),int(z.cross_border.sum()),int(z.is_alerted_transaction.sum())))
  if i%1000==0: print(f"Cases processed: {i:,}/{len(q):,}",flush=True)
 out=pd.concat(chunks,ignore_index=True); sm=pd.DataFrame(stats,columns=["CASE_ID","account_id","network_transactions","unique_counterparties","cross_border_transactions","alerted_transactions"]).sort_values(["unique_counterparties","network_transactions"],ascending=False)
 a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False); sm.to_csv(a.summary,index=False)
 print("\n=== FULL CASE NETWORK MART ===",flush=True); print(f"Network rows: {len(out):,} | Cases: {out.CASE_ID.nunique():,}",flush=True)
 print("\n=== TOP TEST CASES BY NETWORK COMPLEXITY ===",flush=True); print(sm.head(15).to_string(index=False),flush=True); print(f"\nSaved: {a.out}\nSaved: {a.summary}",flush=True)
if __name__=="__main__": main()
