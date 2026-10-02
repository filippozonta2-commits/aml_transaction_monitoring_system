"""Deposit-Send V2: cumulative flow-of-funds analysis on DEVELOPMENT.

For each development Cash Deposit, measures subsequent non-deposit outgoing
activity by the same account over several horizons. TRAIN is warm-up/context
only. HOLDOUT is never accessed.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load(path, period):
    cols=["Time","Date","Sender_account","Receiver_account","Amount",
          "Payment_type","Is_laundering","Laundering_type"]
    d=pd.read_csv(path,usecols=cols)
    d["ts"]=pd.to_datetime(
        pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")
        +" "+d["Time"].astype(str),errors="coerce")
    d["Period"]=period
    return d.sort_values("ts")


def build_features(deposits, outgoing, horizon_hours):
    """Account-wise searchsorted avoids a deposit×transaction Cartesian join."""
    rows=[]
    out_groups={k:g for k,g in outgoing.groupby("Sender_account",sort=False)}
    for i,(acct,gdep) in enumerate(deposits.groupby("Sender_account",sort=False),1):
        gout=out_groups.get(acct)
        if gout is None:
            for r in gdep.itertuples():
                rows.append([r.deposit_id,0,0.0,0,np.nan,0.0])
            continue

        times=gout["ts"].values.astype("datetime64[ns]")
        amounts=gout["Amount"].to_numpy(float)
        receivers=gout["Receiver_account"].to_numpy()
        for r in gdep.itertuples():
            t=np.datetime64(r.ts)
            end=t+np.timedelta64(horizon_hours,"h")
            lo=np.searchsorted(times,t,side="right")
            hi=np.searchsorted(times,end,side="right")
            if hi<=lo:
                rows.append([r.deposit_id,0,0.0,0,np.nan,0.0])
                continue
            am=amounts[lo:hi]
            first=(times[lo]-t)/np.timedelta64(1,"h")
            rows.append([
                r.deposit_id,hi-lo,float(am.sum()),
                int(pd.unique(receivers[lo:hi]).size),
                float(first),float(np.median(am))
            ])
    return pd.DataFrame(rows,columns=[
        "deposit_id","outgoing_count","cumulative_outflow",
        "unique_receivers","hours_to_first_outflow","outgoing_median"
    ])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/temporal/SAML-D_train.csv"))
    p.add_argument("--development",type=Path,default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("results/deposit_send_v2"))
    a=p.parse_args()

    print("Loading TRAIN context + DEVELOPMENT...")
    tr=load(a.train,"train"); dev=load(a.development,"development")
    start=dev["ts"].min()
    context=tr[tr["ts"].ge(start-pd.Timedelta(days=7))]
    data=pd.concat([context,dev],ignore_index=True).sort_values("ts")
    del tr,context
    print("HOLDOUT is not accessed.")

    dep=dev[dev["Payment_type"].eq("Cash Deposit")].copy().reset_index(drop=True)
    dep["deposit_id"]=np.arange(len(dep))
    dep["target"]=(dep["Is_laundering"].eq(1)&dep["Laundering_type"].eq("Deposit-Send"))
    dep=dep.rename(columns={"Amount":"deposit_amount"})
    outgoing=data[~data["Payment_type"].eq("Cash Deposit")][
        ["ts","Sender_account","Receiver_account","Amount"]
    ].copy()

    print(f"Development Cash Deposits: {len(dep):,}")
    print(f"Deposit-Send target deposits: {int(dep['target'].sum()):,}")

    all_results=[]; feature_frames=[]
    for hours in [6,12,24,48,72,168]:
        print(f"Building cumulative flow features: {hours}h...")
        f=build_features(dep[["deposit_id","ts","Sender_account"]],outgoing,hours)
        x=dep[["deposit_id","deposit_amount","target","Is_laundering",
               "Sender_account"]].merge(f,on="deposit_id",how="left")
        x["outflow_ratio"]=x["cumulative_outflow"]/x["deposit_amount"].replace(0,np.nan)
        x["Horizon_hours"]=hours
        feature_frames.append(x)

        for min_count in [1,2,3,5]:
            for lo in [.25,.50,.75,1.00]:
                for hi in [1.25,1.50,2.00,3.00]:
                    for min_receivers in [1,2,3]:
                        m=(x["outgoing_count"].ge(min_count)&
                           x["outflow_ratio"].between(lo,hi)&
                           x["unique_receivers"].ge(min_receivers))
                        trig=int(m.sum()); hits=int((m&x["target"]).sum())
                        aml=int((m&x["Is_laundering"].eq(1)).sum())
                        target=int(x["target"].sum())
                        target_accts=x.loc[x["target"],"Sender_account"].nunique()
                        hit_accts=x.loc[m&x["target"],"Sender_account"].nunique()
                        all_results.append(dict(
                            Horizon_hours=hours,Min_outgoing_count=min_count,
                            Min_outflow_ratio=lo,Max_outflow_ratio=hi,
                            Min_receivers=min_receivers,
                            Triggered_deposits=trig,
                            Trigger_rate=trig/len(x),
                            AML_cases=aml,
                            Precision=aml/trig if trig else np.nan,
                            Deposit_Send_hits=hits,
                            Deposit_Send_recall=hits/target if target else np.nan,
                            Deposit_Send_hit_accounts=hit_accts,
                            Deposit_Send_account_recall=hit_accts/target_accts if target_accts else np.nan
                        ))

    res=pd.DataFrame(all_results)
    eligible=res[res["Trigger_rate"].le(.10)]
    top=eligible.sort_values(
        ["Deposit_Send_recall","Precision","Trigger_rate"],
        ascending=[False,False,True]).head(20)

    a.output_dir.mkdir(parents=True,exist_ok=True)
    res.to_csv(a.output_dir/"deposit_send_v2_grid.csv",index=False)
    top.to_csv(a.output_dir/"deposit_send_v2_shortlist.csv",index=False)
    pd.concat(feature_frames,ignore_index=True).to_csv(
        a.output_dir/"deposit_send_v2_features.csv",index=False)

    print("\n=== DEPOSIT-SEND V2 CUMULATIVE FLOW SHORTLIST ===")
    print(top.to_string(index=False))
    print("\nAll metrics are DEVELOPMENT-only. HOLDOUT was not accessed.")


if __name__=="__main__":
    main()
