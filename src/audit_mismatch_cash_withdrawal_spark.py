"""Audit the suspicious 61/61 AML Cash Withdrawal subset inside cross-border currency mismatch.

DEVELOPMENT only. Checks whether Payment_type=Cash Withdrawal is label-associated
globally, profiles the mismatch subset, and reports overlap/incrementality vs frozen
scenario flags when a compatible unified artifact is available.
"""
from pathlib import Path
import argparse
from pyspark.sql import SparkSession, functions as F

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
    ap.add_argument("--out",default="results/candidate_scenarios/mismatch_cash_withdrawal_audit")
    a=ap.parse_args()

    spark=(SparkSession.builder.appName("audit-mismatch-cash-withdrawal")
           .master("local[*]").config("spark.sql.shuffle.partitions","16").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    x=spark.read.option("header",True).option("inferSchema",True).csv(a.development)
    aml=F.col("Is_laundering").cast("int")==1
    cash=F.lower(F.trim("Payment_type"))=="cash withdrawal"
    mismatch=((F.upper(F.trim("Sender_bank_location"))!=F.upper(F.trim("Receiver_bank_location"))) &
              (F.upper(F.trim("Payment_currency"))!=F.upper(F.trim("Received_currency"))))
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    print("\n=== 1. GLOBAL PAYMENT-TYPE LABEL AUDIT — DEVELOPMENT ONLY ===")
    pay=(x.groupBy("Payment_type")
         .agg(F.count("*").alias("transactions"),
              F.sum(F.when(aml,1).otherwise(0)).alias("aml_hits"))
         .withColumn("aml_rate",F.col("aml_hits")/F.col("transactions"))
         .orderBy(F.desc("aml_rate"),F.desc("transactions")))
    pay.show(100,truncate=False)
    pay.coalesce(1).write.mode("overwrite").option("header",True).csv(str(out/"payment_type_label_audit"))

    cw=x.filter(cash).cache()
    mmcw=x.filter(cash & mismatch).cache()
    print("\n=== 2. CASH WITHDRAWAL AUDIT ===")
    print(f"All Cash Withdrawal rows: {cw.count():,} | AML: {cw.filter(aml).count():,}")
    print(f"Cash Withdrawal + mismatch rows: {mmcw.count():,} | AML: {mmcw.filter(aml).count():,}")

    print("\nLaundering typologies — Cash Withdrawal + mismatch:")
    mmcw.groupBy("Laundering_type").agg(F.count("*").alias("n")).orderBy(F.desc("n")).show(100,False)

    print("\nCountry pairs:")
    mmcw.groupBy("Sender_bank_location","Receiver_bank_location").agg(
        F.count("*").alias("n"),F.sum(F.when(aml,1).otherwise(0)).alias("aml_hits")
    ).orderBy(F.desc("n")).show(100,False)

    print("\nCurrency pairs:")
    mmcw.groupBy("Payment_currency","Received_currency").agg(
        F.count("*").alias("n"),F.sum(F.when(aml,1).otherwise(0)).alias("aml_hits")
    ).orderBy(F.desc("n")).show(100,False)

    print("\nAmount summary:")
    mmcw.select(F.col("Amount").cast("double").alias("Amount")).summary("count","min","25%","50%","75%","max","mean","stddev").show(n=20, truncate=False)

    cols=[c for c in ["Time","Date","Sender_account","Receiver_account","Amount","Payment_currency",
          "Received_currency","Sender_bank_location","Receiver_bank_location","Payment_type",
          "Is_laundering","Laundering_type"] if c in x.columns]
    mmcw.select(*cols).coalesce(1).write.mode("overwrite").option("header",True).csv(str(out/"mismatch_cash_withdrawal_rows"))

    print("\n=== 3. DECOMPOSITION ===")
    for name,cond in [
        ("MISMATCH_CASH_WITHDRAWAL", cash & mismatch),
        ("MISMATCH_NON_CASH", (~cash) & mismatch),
        ("CASH_WITHDRAWAL_ANY", cash),
    ]:
        r=x.filter(cond).agg(F.count("*").alias("n"),
             F.sum(F.when(aml,1).otherwise(0)).alias("hits")).first()
        n=int(r.n or 0); h=int(r.hits or 0)
        print(f"{name}: alerts={n:,} | AML={h:,} | precision={(h/n if n else 0):.4%}")

    print("\nAudit artifacts saved:",a.out)
    print("Interpret 61/61 only after the global Payment_type audit; do not promote from label association alone.")
    print("DEVELOPMENT only. No HOLDOUT accessed.")
    spark.stop()

if __name__=="__main__":
    main()
