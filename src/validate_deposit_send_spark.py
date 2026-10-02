"""Validate Deposit-Send receiver flow features: Pandas vs PySpark.

Checks the same DEVELOPMENT Deposit-Send targets at 72h and 168h and reports
row-level disagreements in outgoing count and cumulative outflow.
HOLDOUT is never accessed.
"""
import argparse
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F

COLS=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
      "Is_laundering","Laundering_type"]

def pandas_read(path):
    d=pd.read_csv(path,usecols=COLS)
    d["ts"]=pd.to_datetime(
        pd.to_datetime(d["Date"],errors="coerce").dt.strftime("%Y-%m-%d")
        +" "+d["Time"].astype(str),errors="coerce")
    d["Amount"]=pd.to_numeric(d["Amount"],errors="coerce")
    return d.sort_values("ts")

def pandas_features(targets, events, hours):
    groups={k:g.sort_values("ts") for k,g in events.groupby("Sender_account",sort=False)}
    rows=[]
    for r in targets.itertuples():
        g=groups.get(r.Receiver_account)
        if g is None:
            rows.append((r.validation_id,0,0.0))
            continue
        z=g[(g["ts"]>r.ts)&(g["ts"]<=r.ts+pd.Timedelta(hours=hours))]
        rows.append((r.validation_id,len(z),float(z["Amount"].sum()) if len(z) else 0.0))
    return pd.DataFrame(rows,columns=["validation_id",f"pd_count_{hours}",f"pd_sum_{hours}"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",default="data/temporal/SAML-D_train.csv")
    ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
    ap.add_argument("--sample-size",type=int,default=25)
    ap.add_argument("--output",default="results/deposit_send_v3_spark/validation_pandas_vs_spark.csv")
    a=ap.parse_args()

    print("Loading Pandas validation data...")
    tr=pandas_read(a.train); dev=pandas_read(a.development)
    start=dev["ts"].min()
    events=pd.concat([tr[tr["ts"]>=start-pd.Timedelta(days=7)],dev],
                     ignore_index=True).sort_values("ts")
    targets=dev[(dev["Is_laundering"]==1)&
                (dev["Laundering_type"]=="Deposit-Send")].copy()
    # Deterministic sample, plus prioritize targets known to change between horizons.
    p72=pandas_features(targets.assign(validation_id=np.arange(len(targets))),events,72)
    tmp=targets.assign(validation_id=np.arange(len(targets)))
    p168=pandas_features(tmp,events,168)
    allp=tmp[["validation_id","ts","Receiver_account","Amount"]].merge(p72,on="validation_id").merge(p168,on="validation_id")
    changing=allp[allp["pd_count_168"]!=allp["pd_count_72"]]
    stable=allp[~allp["validation_id"].isin(changing["validation_id"])]
    sample=pd.concat([changing,stable],ignore_index=True).head(a.sample_size).copy()
    print(f"Targets changing between 72h and 168h in Pandas: {len(changing):,}")
    print(f"Validation sample: {len(sample):,}")

    spark=(SparkSession.builder.appName("AML-Pandas-Spark-Validation")
           .master("local[*]").config("spark.sql.shuffle.partitions","8")
           .config("spark.driver.memory","4g").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    def sr(path):
        d=spark.read.option("header",True).option("inferSchema",True).csv(path).select(*COLS)
        date_raw=F.col("Date").cast("string")
        date_parsed=F.coalesce(
            F.to_date(date_raw,"yyyy-MM-dd"),
            F.to_date(date_raw,"M/d/yyyy"),
            F.to_date(date_raw,"MM/dd/yyyy")
        )
        date_str=F.date_format(date_parsed,"yyyy-MM-dd")
        time_raw=F.col("Time").cast("string")
        ts=F.coalesce(
            F.to_timestamp(time_raw),
            F.to_timestamp(F.concat_ws(" ",date_str,time_raw))
        )
        return d.withColumn("ts",ts).withColumn("Amount",F.col("Amount").cast("double"))

    strn=sr(a.train); sdev=sr(a.development)
    sstart=sdev.agg(F.min("ts")).first()[0]
    if sstart is None:
        raise RuntimeError("Spark parsed all development timestamps as NULL; check Date/Time formats.")
    cutoff=sstart-pd.Timedelta(days=7)
    print(f"Spark development start: {sstart} | context cutoff: {cutoff}")
    sevents=(strn.filter(F.col("ts")>=F.lit(cutoff.to_pydatetime()))
             .unionByName(sdev)
             .select(F.col("Sender_account").alias("event_sender"),
                     F.col("Amount").alias("event_amount"),
                     F.col("ts").alias("event_ts")))

    # Use exact validation IDs/account/timestamps from Pandas sample, avoiding any
    # cross-engine synthetic-ID mismatch.
    ss=spark.createDataFrame(sample[["validation_id","ts","Receiver_account"]])
    for hours in (72,168):
        b=ss.alias("b"); e=sevents.alias("e")
        cond=((F.col("b.Receiver_account")==F.col("e.event_sender"))&
              (F.col("e.event_ts")>F.col("b.ts"))&
              (F.col("e.event_ts")<=F.col("b.ts")+F.expr(f"INTERVAL {hours} HOURS")))
        agg=(b.join(e,cond,"left").groupBy("b.validation_id")
             .agg(F.count("e.event_ts").alias(f"sp_count_{hours}"),
                  F.coalesce(F.sum("e.event_amount"),F.lit(0.0)).alias(f"sp_sum_{hours}")))
        ss=ss.join(agg,"validation_id","left")

    sp=ss.toPandas()
    out=sample.merge(sp[["validation_id","sp_count_72","sp_sum_72",
                         "sp_count_168","sp_sum_168"]],on="validation_id")
    for h in (72,168):
        out[f"count_match_{h}"]=out[f"pd_count_{h}"]==out[f"sp_count_{h}"]
        out[f"sum_match_{h}"]=np.isclose(out[f"pd_sum_{h}"],out[f"sp_sum_{h}"],
                                         rtol=1e-9,atol=1e-6)
    out["all_match"]=out[[f"count_match_{h}" for h in (72,168)]+
                         [f"sum_match_{h}" for h in (72,168)]].all(axis=1)
    out.to_csv(a.output,index=False)

    print("\n=== PANDAS VS PYSPARK VALIDATION ===")
    print(out[["validation_id","pd_count_72","sp_count_72","pd_count_168",
               "sp_count_168","pd_sum_72","sp_sum_72","pd_sum_168",
               "sp_sum_168","all_match"]].to_string(index=False))
    print(f"\nRows fully matching: {int(out['all_match'].sum())}/{len(out)}")
    print(f"Saved: {a.output}")
    print("HOLDOUT was not accessed.")
    spark.stop()

if __name__=="__main__":
    main()
