"""Audit transaction-country normalization against country-risk reference.
DEVELOPMENT only; no scenario thresholds selected and no HOLDOUT access.
"""
import argparse
from pyspark.sql import SparkSession, functions as F

def main():
 p=argparse.ArgumentParser(); p.add_argument("--development",default="data/temporal/SAML-D_development.csv"); p.add_argument("--country-risk",default="data/country_risk.csv"); a=p.parse_args()
 spark=SparkSession.builder.appName("aml-geo-reference-audit").getOrCreate(); spark.sparkContext.setLogLevel("WARN")
 x=spark.read.option("header",True).option("inferSchema",True).csv(a.development)
 r=spark.read.option("header",True).option("inferSchema",True).csv(a.country_risk)
 vals=(x.select(F.col("Sender_bank_location").cast("string").alias("loc")).union(x.select(F.col("Receiver_bank_location").cast("string").alias("loc")))
       .groupBy("loc").count().orderBy(F.desc("count")))
 print("\n=== TOP TRANSACTION LOCATION VALUES ==="); vals.show(40,False)
 print("\n=== COUNTRY REFERENCE SAMPLE ==="); r.select("Country","ISO Code","Overall Score","ML/TF Risk","Sanction").show(40,False)
 print("\n=== SANCTION VALUE DISTRIBUTION ==="); r.groupBy("Sanction").count().orderBy(F.desc("count")).show(50,False)
 # Test three plausible keys independently instead of assuming the source representation.
 ref=r.select(F.upper(F.trim(F.col("ISO Code").cast("string"))).alias("iso"),
              F.upper(F.trim(F.col("Country").cast("string"))).alias("country")).dropDuplicates()
 distinct=vals.select(F.upper(F.trim("loc")).alias("loc"),"count")
 iso=distinct.join(ref,distinct.loc==ref.iso,"left")
 country=distinct.join(ref,distinct.loc==ref.country,"left")
 total=distinct.agg(F.sum("count")).first()[0]
 iso_hit=iso.filter(F.col("iso").isNotNull()).agg(F.sum("count")).first()[0] or 0
 country_hit=country.filter(F.col("country").isNotNull()).agg(F.sum("count")).first()[0] or 0
 print(f"\nWeighted direct ISO match coverage: {iso_hit/total:.2%}")
 print(f"Weighted direct country-name match coverage: {country_hit/total:.2%}")
 print("\n=== UNMATCHED TOP VALUES (ISO attempt) ===")
 iso.filter(F.col("iso").isNull()).select("loc","count").orderBy(F.desc("count")).show(50,False)
 print("No HOLDOUT accessed."); spark.stop()
if __name__=="__main__": main()
