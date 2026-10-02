"""Segment cross-border currency mismatch on DEVELOPMENT using PySpark.

Profiles Payment_type, country pair, currency pair, and selected combinations.
Reports trigger volume, AML hits, precision, recall and lift. No HOLDOUT access.
"""
from pathlib import Path
import argparse, pandas as pd
from pyspark.sql import SparkSession, functions as F

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
 ap.add_argument("--out-dir",default="results/candidate_scenarios/cross_border_segments")
 a=ap.parse_args()
 spark=(SparkSession.builder.appName("aml-cross-border-segmentation").master("local[*]")
        .config("spark.sql.shuffle.partitions","16").getOrCreate())
 spark.sparkContext.setLogLevel("WARN")
 x=spark.read.option("header",True).option("inferSchema",True).csv(a.development)
 aml=F.col("Is_laundering").cast("int")==1
 mismatch=((F.upper(F.trim("Sender_bank_location"))!=F.upper(F.trim("Receiver_bank_location"))) &
           (F.upper(F.trim("Payment_currency"))!=F.upper(F.trim("Received_currency"))))
 m=x.filter(mismatch).cache()
 total=x.count(); positives=x.filter(aml).count(); prevalence=positives/total
 out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)

 specs={
  "payment_type":["Payment_type"],
  "country_pair":["Sender_bank_location","Receiver_bank_location"],
  "currency_pair":["Payment_currency","Received_currency"],
  "payment_currency_pair":["Payment_type","Payment_currency","Received_currency"],
  "payment_country_pair":["Payment_type","Sender_bank_location","Receiver_bank_location"]
 }
 print("\n=== CROSS-BORDER MISMATCH SEGMENTATION — DEVELOPMENT ONLY ===")
 print(f"Mismatch rows: {m.count():,} | AML hits: {m.filter(aml).count():,}")
 for name,cols in specs.items():
  z=(m.groupBy(*cols).agg(F.count("*").alias("triggered"),
       F.sum(F.when(aml,1).otherwise(0)).alias("aml_hits"))
       .withColumn("precision",F.col("aml_hits")/F.col("triggered"))
       .withColumn("recall",F.col("aml_hits")/F.lit(positives))
       .withColumn("lift_vs_base",F.col("precision")/F.lit(prevalence))
       .orderBy(F.desc("aml_hits"),F.desc("precision")))
  z.coalesce(1).write.mode("overwrite").option("header",True).csv(str(out/name))
  print(f"\n--- {name.upper()} (top 20 by AML hits) ---")
  z.show(20,truncate=False)
 print("\nSaved:",out)
 print("Segmentation diagnostic only. No threshold frozen. No HOLDOUT accessed.")
 spark.stop()
if __name__=="__main__":main()
