"""Materialize frozen Deposit-Send V3 on HOLDOUT. Validation only; no tuning."""
import argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--holdout",default="data/temporal/SAML-D_holdout.csv")
 p.add_argument("--out",default="results/holdout_v3/deposit_send")
 a=p.parse_args()
 spark=(SparkSession.builder.appName("AML-V3-Holdout-Deposit-Send").master("local[*]")
        .config("spark.sql.shuffle.partitions","16").config("spark.driver.memory","4g").getOrCreate())
 spark.sparkContext.setLogLevel("WARN")
 cols=["Time","Date","Sender_account","Receiver_account","Amount","Payment_type","Is_laundering","Laundering_type"]
 d=spark.read.option("header",True).option("inferSchema",False).csv(a.holdout).select(*cols)
 raw=F.trim(F.col("Date")); date=F.coalesce(F.to_date(raw,"yyyy-MM-dd"),F.to_date(raw,"M/d/yyyy"),F.to_date(raw,"MM/dd/yyyy"),F.to_date(raw,"yyyy/MM/dd"))
 dev=(d.withColumn("ts",F.to_timestamp(F.concat_ws(" ",F.date_format(date,"yyyy-MM-dd"),F.trim(F.col("Time")))))
      .withColumn("Sender_account",F.col("Sender_account").cast("long")).withColumn("Receiver_account",F.col("Receiver_account").cast("long"))
      .withColumn("Amount",F.col("Amount").cast("double")).withColumn("Is_laundering",F.col("Is_laundering").cast("int"))).cache()
 n=dev.count(); schema=dev.schema.add("holdout_row_id","long",False)
 idx=spark.createDataFrame(dev.rdd.zipWithIndex().map(lambda z:tuple(z[0])+(z[1],)),schema).cache(); idx.count()
 events=dev.select(F.col("Sender_account").alias("event_sender"),F.col("Amount").alias("event_amount"),
   F.col("Payment_type").alias("event_payment_type"),F.col("ts").alias("event_ts")).repartition(16,"event_sender").cache()
 b=idx.select("holdout_row_id","ts","Receiver_account","Amount","Is_laundering","Laundering_type").alias("b"); e=events.alias("e")
 cond=((F.col("b.Receiver_account")==F.col("e.event_sender"))&(F.col("e.event_ts")>F.col("b.ts"))&
       (F.col("e.event_ts")<=F.col("b.ts")+F.expr("INTERVAL 72 HOURS")))
 feat=(b.join(e,cond,"left").groupBy("b.holdout_row_id","b.Amount","b.Is_laundering","b.Laundering_type")
   .agg(F.count("e.event_ts").alias("outgoing_count"),F.coalesce(F.sum("e.event_amount"),F.lit(0.0)).alias("cumulative_outflow"),
        F.sum(F.when(F.col("e.event_payment_type")=="Cross-border",1).otherwise(0)).alias("cross_border_count"))
   .withColumn("outflow_ratio",F.when(F.col("Amount")!=0,F.col("cumulative_outflow")/F.col("Amount"))))
 rule=(F.col("outgoing_count")>=1)&F.col("outflow_ratio").between(.50,3.0)&(F.col("cross_border_count")>=1)
 out=feat.withColumn("SCN_DEPOSIT_SEND",rule.cast("int")).select("holdout_row_id","SCN_DEPOSIT_SEND").orderBy("holdout_row_id")
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 out.coalesce(1).write.mode("overwrite").option("header",True).csv(a.out)
 print("=== FROZEN V3 DEPOSIT-SEND — HOLDOUT ===")
 print(f"Rows: {n:,} | alerts: {out.filter('SCN_DEPOSIT_SEND=1').count():,}")
 print("No threshold tuning performed. Saved:",a.out); spark.stop()
if __name__=="__main__": main()
