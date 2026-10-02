"""Materialize the frozen Deposit-Send scenario on DEVELOPMENT with PySpark.

Writes one compact mapping of development_row_id -> SCN_DEPOSIT_SEND and
reconciles the output against the frozen DEVELOPMENT metrics. TRAIN supplies
only the 72-hour warm-up context. HOLDOUT is never read.
"""
from pathlib import Path
import argparse
from pyspark.sql import SparkSession, functions as F

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",default="data/temporal/SAML-D_train.csv")
    p.add_argument("--development",default="data/temporal/SAML-D_development.csv")
    p.add_argument("--output-dir",default="results/unified_dashboard/deposit_send_flags")
    p.add_argument("--shuffle-partitions",type=int,default=16)
    a=p.parse_args()

    spark=(SparkSession.builder.appName("AML-Materialize-Deposit-Send")
           .master("local[*]")
           .config("spark.sql.shuffle.partitions",str(a.shuffle_partitions))
           .config("spark.driver.memory","4g").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    cols=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
          "Is_laundering","Laundering_type"]

    def read(path):
        d=spark.read.option("header",True).option("inferSchema",False).csv(path).select(*cols)
        raw=F.trim(F.col("Date"))
        date=F.coalesce(F.to_date(raw,"yyyy-MM-dd"),F.to_date(raw,"M/d/yyyy"),
                        F.to_date(raw,"MM/dd/yyyy"),F.to_date(raw,"yyyy/MM/dd"))
        ts=F.to_timestamp(F.concat_ws(" ",F.date_format(date,"yyyy-MM-dd"),F.trim(F.col("Time"))))
        return (d.withColumn("ts",ts)
                 .withColumn("Sender_account",F.col("Sender_account").cast("long"))
                 .withColumn("Receiver_account",F.col("Receiver_account").cast("long"))
                 .withColumn("Amount",F.col("Amount").cast("double"))
                 .withColumn("Is_laundering",F.col("Is_laundering").cast("int")))

    print("Loading TRAIN warm-up + DEVELOPMENT with Spark...")
    tr=read(a.train)
    dev=read(a.development).cache()
    n=dev.count(); start=dev.agg(F.min("ts")).first()[0]
    if start is None: raise RuntimeError("Development timestamps parsed as NULL.")
    print(f"Development rows: {n:,} | start: {start}")
    print("HOLDOUT is not accessed.")

    # zipWithIndex preserves CSV row order and therefore matches the Pandas
    # development_row_id used by materialize_frozen_scenarios.py.
    schema=dev.schema.add("development_row_id","long",False)
    indexed=spark.createDataFrame(
        dev.rdd.zipWithIndex().map(lambda z: tuple(z[0])+ (z[1],)), schema
    ).cache()
    indexed.count()

    context=tr.filter(F.col("ts") >= F.lit(start)-F.expr("INTERVAL 72 HOURS"))
    events=context.unionByName(dev).select(
        F.col("Sender_account").alias("event_sender"),
        F.col("Receiver_account").alias("event_receiver"),
        F.col("Amount").alias("event_amount"),
        F.col("Payment_type").alias("event_payment_type"),
        F.col("ts").alias("event_ts")
    ).repartition(a.shuffle_partitions,"event_sender").cache()

    b=indexed.select("development_row_id","ts","Receiver_account","Amount",
                     "Is_laundering","Laundering_type").alias("b")
    e=events.alias("e")
    cond=((F.col("b.Receiver_account")==F.col("e.event_sender")) &
          (F.col("e.event_ts")>F.col("b.ts")) &
          (F.col("e.event_ts")<=F.col("b.ts")+F.expr("INTERVAL 72 HOURS")))
    feat=(b.join(e,cond,"left")
          .groupBy("b.development_row_id","b.Amount","b.Is_laundering","b.Laundering_type")
          .agg(F.count("e.event_ts").alias("outgoing_count"),
               F.coalesce(F.sum("e.event_amount"),F.lit(0.0)).alias("cumulative_outflow"),
               F.sum(F.when(F.col("e.event_payment_type")=="Cross-border",1).otherwise(0))
                .alias("cross_border_count"))
          .withColumn("outflow_ratio",
              F.when(F.col("Amount")!=0,F.col("cumulative_outflow")/F.col("Amount"))))

    rule=((F.col("outgoing_count")>=1)&F.col("outflow_ratio").between(.50,3.0)&
          (F.col("cross_border_count")>=1))
    flags=(feat.withColumn("SCN_DEPOSIT_SEND",F.when(rule,1).otherwise(0))
           .select("development_row_id","Is_laundering","Laundering_type","SCN_DEPOSIT_SEND")
           .cache())
    flags.count()

    target=(F.col("Is_laundering")==1)&(F.col("Laundering_type")=="Deposit-Send")
    m=flags.agg(
        F.sum("SCN_DEPOSIT_SEND").alias("triggered"),
        F.sum(F.when((F.col("SCN_DEPOSIT_SEND")==1)&(F.col("Is_laundering")==1),1).otherwise(0)).alias("aml"),
        F.sum(F.when((F.col("SCN_DEPOSIT_SEND")==1)&target,1).otherwise(0)).alias("hits"),
        F.sum(F.when(target,1).otherwise(0)).alias("targets")
    ).first()
    trig=int(m.triggered or 0); aml=int(m.aml or 0); hits=int(m.hits or 0); targets=int(m.targets or 0)

    out=Path(a.output_dir)
    (flags.select("development_row_id","SCN_DEPOSIT_SEND").coalesce(1)
     .write.mode("overwrite").option("header",True).csv(str(out)))

    print("\n=== MATERIALIZED DEPOSIT-SEND FREEZE ===")
    print(f"Triggered: {trig:,}")
    print(f"Trigger rate: {trig/n:.6f}")
    print(f"AML cases: {aml:,}")
    print(f"Precision: {aml/trig if trig else 0:.6f}")
    print(f"Deposit-Send hits: {hits:,}/{targets:,}")
    print(f"Deposit-Send recall: {hits/targets if targets else 0:.6f}")
    print("\nFrozen target: Triggered=24,168 | AML_cases=107 | Precision=0.004427 | hits=46/175 | recall=0.262857")
    print(f"Saved: {out}")
    print("Development-only. HOLDOUT was not accessed.")
    spark.stop()

if __name__=="__main__":
    main()
