"""Tune cross-border currency-mismatch candidates on DEVELOPMENT with PySpark.

Benchmarks absolute amount quantiles plus account-relative amount thresholds.
Diagnostic only: does not freeze a threshold and never accesses HOLDOUT.
"""
from pathlib import Path
import argparse, pandas as pd
from pyspark.sql import SparkSession, functions as F, Window

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
 ap.add_argument("--out",default="results/candidate_scenarios/cross_border_mismatch_benchmark.csv")
 a=ap.parse_args()

 spark=(SparkSession.builder.appName("aml-cross-border-mismatch-benchmark")
        .master("local[*]").config("spark.sql.shuffle.partitions","16").getOrCreate())
 spark.sparkContext.setLogLevel("WARN")
 x=spark.read.option("header",True).option("inferSchema",True).csv(a.development)
 x=x.withColumn("Amount",F.col("Amount").cast("double"))
 aml=F.col("Is_laundering").cast("int")==1

 # Semantic guard: cash deposit/withdrawal do not have a true two-sided counterparty pair.
 # Keep raw sender/receiver fields for lineage, but exclude cash transactions from
 # cross-border/currency-pair scenario interpretation.
 pt=F.lower(F.trim(F.col("Payment_type")))
 pair_applicable=~pt.isin("cash withdrawal","cash deposit")
 base=(pair_applicable &
       (F.upper(F.trim("Sender_bank_location"))!=F.upper(F.trim("Receiver_bank_location"))) &
       (F.upper(F.trim("Payment_currency"))!=F.upper(F.trim("Received_currency"))))
 xb=x.filter(base).cache()
 base_n=xb.count()
 print(f"Semantically valid base mismatch triggers: {base_n:,}")

 qs=[0.50,0.75,0.90,0.95,0.975,0.99]
 vals=x.approxQuantile("Amount",qs,0.001)
 rows=[]
 def metric(name,df):
  z=df.agg(F.count("*").alias("n"),F.sum(F.when(aml,1).otherwise(0)).alias("hits")).first()
  n=int(z.n or 0); h=int(z.hits or 0)
  rows.append((name,n,h,h/n if n else 0))

 metric("MISMATCH_PURE",xb)
 for q,v in zip(qs,vals):
  metric(f"MISMATCH_AMOUNT_Q{q}",xb.filter(F.col("Amount")>=F.lit(v)))

 # Account-relative benchmark: compare amount to sender's historical distribution
 # using only preceding DEVELOPMENT rows; no future-row leakage.
 ts=F.to_timestamp(F.concat_ws(" ",F.col("Date"),F.col("Time")))
 z=x.withColumn("_ts",ts)
 w=(Window.partitionBy("Sender_account").orderBy(F.col("_ts").cast("long"))
    .rowsBetween(Window.unboundedPreceding,-1))
 z=(z.withColumn("_hist_avg",F.avg("Amount").over(w))
      .withColumn("_hist_sd",F.stddev_samp("Amount").over(w))
      .withColumn("_hist_n",F.count("Amount").over(w)))
 zb=z.filter(base & (F.col("_hist_n")>=10)).cache()
 for k in [2.0,3.0,4.0,5.0]:
  metric(f"MISMATCH_ACCOUNT_Z{k:g}",
         zb.filter((F.col("_hist_sd")>0)&
                   (F.col("Amount")>=F.col("_hist_avg")+F.lit(k)*F.col("_hist_sd"))))

 out=pd.DataFrame(rows,columns=["candidate","triggered","aml_hits","precision"])
 total=x.count(); positives=x.filter(aml).count(); prev=positives/total
 out["trigger_rate"]=out.triggered/total
 out["recall"]=out.aml_hits/positives
 out["lift_vs_base"]=out.precision/prev
 out=out.sort_values(["aml_hits","triggered"],ascending=[False,True])
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.out,index=False)
 print("\n=== CROSS-BORDER CURRENCY MISMATCH BENCHMARK — DEVELOPMENT ONLY ===")
 print(f"Transactions: {total:,} | AML positives: {positives:,} | prevalence: {prev:.4%}")
 print(out.to_string(index=False))
 print("\nAbsolute quantiles are DEVELOPMENT-wide candidate thresholds.")
 print("Account-relative variants use prior sender history only (>=10 prior transactions).")
 print("No threshold frozen. No HOLDOUT accessed.")
 print("Saved:",a.out)
 spark.stop()

if __name__=="__main__":main()
