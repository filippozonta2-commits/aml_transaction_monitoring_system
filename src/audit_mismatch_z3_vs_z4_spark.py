"""Audit the incremental AML cases captured by mismatch Z3 but not Z4.

DEVELOPMENT only. Uses the same semantic cash guard and prior-sender-history
definition as evaluate_cross_border_mismatch_incremental_spark.py.
"""
from pathlib import Path
import argparse
from pyspark.sql import SparkSession, functions as F, Window

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
    ap.add_argument("--frozen-non-ds",default="results/unified_dashboard/materialized_non_deposit_send.csv")
    ap.add_argument("--frozen-ds",default="results/unified_dashboard/deposit_send_flags")
    ap.add_argument("--out",default="results/candidate_scenarios/mismatch_z3_vs_z4_audit")
    a=ap.parse_args()

    spark=(SparkSession.builder.appName("aml-mismatch-z3-z4-audit")
           .master("local[*]").config("spark.sql.shuffle.partitions","16").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    x=(spark.read.option("header",True).option("inferSchema",True).csv(a.development)
       .withColumn("Amount",F.col("Amount").cast("double")))
    schema=x.schema
    x=spark.createDataFrame(x.rdd.zipWithIndex().map(lambda ri: tuple(ri[0])+(ri[1],)),
                            schema.add("development_row_id","long"))

    fr=(spark.read.option("header",True).option("inferSchema",True).csv(a.frozen_non_ds)
        .select("development_row_id","SCN_STRUCTURING","SCN_FAN_OUT","SCN_FAN_IN",
                "SCN_CASH_WITHDRAWAL","SCN_SMURFING"))
    ds=(spark.read.option("header",True).option("inferSchema",True).csv(a.frozen_ds)
        .select("development_row_id","SCN_DEPOSIT_SEND"))
    frozen=["SCN_STRUCTURING","SCN_DEPOSIT_SEND","SCN_FAN_OUT","SCN_FAN_IN",
            "SCN_CASH_WITHDRAWAL","SCN_SMURFING"]
    fr=fr.join(ds,"development_row_id","inner").withColumn(
        "frozen_alert",
        F.greatest(*[F.coalesce(F.col(c).cast("int"),F.lit(0)) for c in frozen])
    ).select("development_row_id","frozen_alert")

    ts=F.to_timestamp(F.concat_ws(" ",F.col("Date"),F.col("Time")))
    w=(Window.partitionBy("Sender_account").orderBy(F.col("_ts").cast("long"))
       .rowsBetween(Window.unboundedPreceding,-1))
    z=(x.withColumn("_ts",ts)
       .withColumn("_hist_avg",F.avg("Amount").over(w))
       .withColumn("_hist_sd",F.stddev_samp("Amount").over(w))
       .withColumn("_hist_n",F.count("Amount").over(w))
       .withColumn("amount_zscore",
           F.when((F.col("_hist_n")>=10)&(F.col("_hist_sd")>0),
                  (F.col("Amount")-F.col("_hist_avg"))/F.col("_hist_sd"))))

    pt=F.lower(F.trim(F.col("Payment_type")))
    base=((~pt.isin("cash withdrawal","cash deposit")) &
          (F.upper(F.trim(F.col("Sender_bank_location"))) != F.upper(F.trim(F.col("Receiver_bank_location")))) &
          (F.upper(F.trim(F.col("Payment_currency"))) != F.upper(F.trim(F.col("Received_currency")))))
    z=(z.withColumn("MISMATCH_Z3",(base&(F.col("amount_zscore")>=3)).cast("int"))
         .withColumn("MISMATCH_Z4",(base&(F.col("amount_zscore")>=4)).cast("int"))
         .join(fr,"development_row_id","inner"))

    lost=(z.filter((F.col("MISMATCH_Z3")==1)&(F.col("MISMATCH_Z4")==0)&
                   (F.col("frozen_alert")==0)&(F.col("Is_laundering")==1))
          .select("development_row_id","Date","Time","Sender_account","Receiver_account",
                  "Amount","amount_zscore","_hist_n","_hist_avg","_hist_sd",
                  "Payment_type","Sender_bank_location","Receiver_bank_location",
                  "Payment_currency","Received_currency","Laundering_type")
          .cache())

    n=lost.count()
    print("\n=== MISMATCH Z3 vs Z4 — AML CASES LOST BY Z4 ===")
    print(f"Incremental AML cases captured by Z3 but not Z4: {n}")
    if n:
        print("\nTypology coverage:")
        lost.groupBy("Laundering_type").count().orderBy(F.desc("count")).show(100,False)
        print("\nCase detail:")
        lost.orderBy("amount_zscore").show(100,False)

    out=Path(a.out)
    lost.write.mode("overwrite").option("header",True).csv(str(out/"cases"))
    lost.groupBy("Laundering_type").count().write.mode("overwrite").option("header",True).csv(str(out/"typologies"))
    print("\nSemantic cash guard active. Prior sender history only.")
    print("DEVELOPMENT only. No threshold frozen. No HOLDOUT accessed.")
    print("Saved:",a.out)
    spark.stop()

if __name__=="__main__":
    main()
