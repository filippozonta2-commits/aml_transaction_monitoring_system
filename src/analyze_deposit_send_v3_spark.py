"""Deposit-Send V3 (PySpark): scalable receiver-based flow-of-funds candidates.

Uses Spark range joins and aggregation instead of per-row Pandas filtering.
Development-only calibration; HOLDOUT is never accessed.
"""
from pathlib import Path
import argparse

from pyspark.sql import SparkSession, functions as F


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",default="data/temporal/SAML-D_train.csv")
    p.add_argument("--development",default="data/temporal/SAML-D_development.csv")
    p.add_argument("--output-dir",default="results/deposit_send_v3_spark")
    p.add_argument("--shuffle-partitions",type=int,default=16)
    a=p.parse_args()

    spark=(SparkSession.builder
           .appName("AML-Deposit-Send-V3")
           .master("local[*]")
           .config("spark.sql.shuffle.partitions",str(a.shuffle_partitions))
           .config("spark.driver.memory","4g")
           .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    cols=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type",
          "Is_laundering","Laundering_type"]

    def read(path):
        d=spark.read.option("header",True).option("inferSchema",True).csv(path).select(*cols)
        # SAML-D Time may already contain a full timestamp or only a time.
        d=d.withColumn(
            "ts",
            F.coalesce(
                F.to_timestamp("Time"),
                F.to_timestamp(F.concat_ws(" ",F.col("Date").cast("string"),
                                             F.col("Time").cast("string")))
            )
        )
        return d.withColumn("Amount",F.col("Amount").cast("double"))

    print("Loading TRAIN + DEVELOPMENT with Spark...")
    tr=read(a.train)
    dev=read(a.development).cache()
    dev_count=dev.count()
    start=dev.agg(F.min("ts")).first()[0]
    print(f"Development rows: {dev_count:,}")
    print("HOLDOUT is not accessed.")

    # Seven-day TRAIN context is sufficient for our longest (168h) horizon.
    context=tr.filter(F.col("ts") >= F.lit(start)-F.expr("INTERVAL 7 DAYS"))
    events=context.unionByName(dev).select(
        F.col("Sender_account").alias("event_sender"),
        F.col("Receiver_account").alias("event_receiver"),
        F.col("Amount").alias("event_amount"),
        F.col("Payment_type").alias("event_payment_type"),
        F.col("ts").alias("event_ts")
    ).repartition(a.shuffle_partitions,"event_sender").cache()

    # Stable synthetic transaction id for this run.
    base=(dev.withColumn("tx_id",F.monotonically_increasing_id())
          .withColumn("target",
              (F.col("Is_laundering")==1)&(F.col("Laundering_type")=="Deposit-Send"))
          .select("tx_id","ts","Receiver_account","Amount","Is_laundering","target")
          .repartition(a.shuffle_partitions,"Receiver_account")
          .cache())

    target_n=base.filter("target").count()
    target_accts=base.filter("target").select("Receiver_account").distinct().count()
    print(f"Deposit-Send targets: {target_n:,} | target receiver accounts: {target_accts:,}")

    configs=[
        (72,.50,1.50,False),(72,.50,2.00,False),(72,.25,2.00,False),
        (72,.50,3.00,False),(168,.50,1.50,False),(168,.50,2.00,False),
        (168,.25,2.00,False),(168,.50,3.00,False),
        (72,.25,3.00,True),(72,.50,3.00,True),
        (168,.25,3.00,True),(168,.50,3.00,True),
    ]

    summaries=[]
    out=Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    for hours in (72,168):
        print(f"Spark receiver range join + aggregation: {hours}h...")
        b=base.alias("b")
        e=events.alias("e")
        cond=(
            (F.col("b.Receiver_account")==F.col("e.event_sender")) &
            (F.col("e.event_ts")>F.col("b.ts")) &
            (F.col("e.event_ts")<=F.col("b.ts")+F.expr(f"INTERVAL {hours} HOURS"))
        )
        joined=b.join(e,cond,"left")
        feat=(joined.groupBy(
                    "b.tx_id","b.Amount","b.Is_laundering","b.target","b.Receiver_account")
              .agg(
                  F.count("e.event_ts").alias("outgoing_count"),
                  F.coalesce(F.sum("e.event_amount"),F.lit(0.0)).alias("cumulative_outflow"),
                  F.countDistinct("e.event_receiver").alias("unique_receivers"),
                  F.sum(F.when(F.col("e.event_payment_type")=="Cross-border",1).otherwise(0))
                   .alias("cross_border_count")
              )
              .withColumn("outflow_ratio",
                  F.when(F.col("Amount")!=0,F.col("cumulative_outflow")/F.col("Amount")))
              .cache())
        feat.count()

        # Only target feature rows are small and useful as auditable diagnostics.
        (feat.filter("target").coalesce(1)
             .write.mode("overwrite").option("header",True)
             .csv(str(out/f"target_features_{hours}h")))

        for h,lo,hi,require_cb in [c for c in configs if c[0]==hours]:
            m=(F.col("outgoing_count")>=1)&F.col("outflow_ratio").between(lo,hi)
            if require_cb:
                m=m&(F.col("cross_border_count")>=1)
            metrics=(feat.agg(
                F.sum(F.when(m,1).otherwise(0)).alias("triggered"),
                F.sum(F.when(m & F.col("target"),1).otherwise(0)).alias("hits"),
                F.sum(F.when(m & (F.col("Is_laundering")==1),1).otherwise(0)).alias("aml"),
                F.countDistinct(F.when(m & F.col("target"),F.col("Receiver_account")))
                 .alias("hit_accounts")
            ).first())
            trig=int(metrics["triggered"] or 0)
            hits=int(metrics["hits"] or 0)
            aml=int(metrics["aml"] or 0)
            ha=int(metrics["hit_accounts"] or 0)
            summaries.append((hours,lo,hi,require_cb,trig,trig/dev_count,
                              aml,aml/trig if trig else None,hits,
                              hits/target_n if target_n else None,ha,
                              ha/target_accts if target_accts else None))
        feat.unpersist()

    schema=["Horizon_hours","Min_outflow_ratio","Max_outflow_ratio",
            "Require_cross_border","Triggered_transactions","Trigger_rate",
            "AML_cases","Precision","Deposit_Send_hits","Deposit_Send_recall",
            "Deposit_Send_hit_accounts","Deposit_Send_account_recall"]
    result=spark.createDataFrame(summaries,schema).orderBy(
        F.desc("Deposit_Send_recall"),F.desc("Precision"),F.asc("Trigger_rate"))
    print("\n=== DEPOSIT-SEND V3 PYSPARK CANDIDATES ===")
    result.show(50,truncate=False)
    result.coalesce(1).write.mode("overwrite").option("header",True).csv(
        str(out/"candidates"))
    print("Development-only. HOLDOUT was not accessed.")
    spark.stop()

if __name__=="__main__":
    main()
