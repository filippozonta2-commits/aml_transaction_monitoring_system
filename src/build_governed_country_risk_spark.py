"""Build governed AML country-risk reference from the project reference table.

Policy layer, not model calibration. DEVELOPMENT/HOLDOUT transactions are not read.
HIGH is sourced from explicit risk indicators in the reference (FATF/ML-TF fields);
SANCTIONED remains a separate dimension.
"""
import argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

def nonempty(c):
 return c.isNotNull() & (F.length(F.trim(c.cast("string")))>0)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--country-risk",default="data/country_risk.csv")
 p.add_argument("--out",default="results/governance/country_risk")
 a=p.parse_args()
 spark=SparkSession.builder.appName("aml-governed-country-risk").getOrCreate()
 spark.sparkContext.setLogLevel("WARN")
 r=spark.read.option("header",True).option("inferSchema",True).csv(a.country_risk)
 # Keep provenance. Do not infer HIGH from a percentile of transaction outcomes.
 fatf=F.lower(F.coalesce(F.col("FATF").cast("string"),F.lit("")))
 mltf=F.lower(F.coalesce(F.col("ML/TF Risk").cast("string"),F.lit("")))
 sanctioned=nonempty(F.col("Sanction"))
 # Explicit reference indicators only. Review generated list before production use.
 high=(fatf.rlike("black|grey|gray|increased monitoring|call for action|high") |
       mltf.rlike("high|very high"))
 elevated=(F.col("Overall Score").cast("double")>=F.lit(6.0))
 tier=(F.when(high,"HIGH").when(elevated,"ELEVATED").otherwise("STANDARD"))
 out=(r.select(
   F.col("Country").alias("country"),
   F.upper(F.trim(F.col("ISO Code").cast("string"))).alias("iso2"),
   F.col("Overall Score").cast("double").alias("overall_score"),
   F.col("ML/TF Risk").cast("string").alias("ml_tf_risk_source"),
   F.col("FATF").cast("string").alias("fatf_source"),
   F.col("Sanction").cast("string").alias("sanction_source"),
   tier.alias("risk_tier"),
   sanctioned.cast("int").alias("is_sanctioned"))
   .dropDuplicates(["iso2"]).orderBy("iso2"))
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 out.coalesce(1).write.mode("overwrite").option("header",True).csv(a.out)
 print("\n=== GOVERNED COUNTRY-RISK LAYER ===")
 out.groupBy("risk_tier").count().orderBy("risk_tier").show(truncate=False)
 print("Sanctioned jurisdictions:",out.filter("is_sanctioned=1").count())
 print("\nHIGH jurisdictions (review before production use):")
 out.filter("risk_tier='HIGH'").select("country","iso2","fatf_source","ml_tf_risk_source").show(200,False)
 print("Saved:",a.out)
 print("Policy/reference artifact only. No DEVELOPMENT labels or HOLDOUT accessed.")
 spark.stop()
if __name__=="__main__": main()
