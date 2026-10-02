"""Materialize policy-driven geography scenarios on DEVELOPMENT.

Uses the governed country-risk registry. No label-based threshold selection.
HOLDOUT is never accessed.

Important: development_row_id must match the Pandas candidate/frozen artifacts.
We therefore assign it with RDD zipWithIndex() over the DEVELOPMENT CSV row order,
the same convention used by materialize_deposit_send_spark.py.
"""
import argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--development",default="data/temporal/SAML-D_development.csv")
 p.add_argument("--governance",default="results/governance/country_risk")
 p.add_argument("--out",default="results/candidate_scenarios/governed_geo_spark")
 a=p.parse_args()

 spark=(SparkSession.builder.appName("aml-governed-geo-scenarios")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions","16").getOrCreate())
 spark.sparkContext.setLogLevel("WARN")

 x=spark.read.option("header",True).option("inferSchema",True).csv(a.development)
 # Stable row id aligned with Pandas np.arange(len(development)).
 schema=x.schema.add("development_row_id","long",False)
 x=spark.createDataFrame(
     x.rdd.zipWithIndex().map(lambda z: tuple(z[0])+(z[1],)),
     schema
 ).cache()
 n=x.count()

 g=spark.read.option("header",True).option("inferSchema",True).csv(a.governance)

 # Map both country names and ISO-2 plus common aliases.
 names=g.select(F.upper(F.trim("country")).alias("raw"),F.upper(F.trim("iso2")).alias("iso2"))
 iso=g.select(F.upper(F.trim("iso2")).alias("raw"),F.upper(F.trim("iso2")).alias("iso2"))
 aliases=spark.createDataFrame([("UK","GB"),("USA","US"),("UAE","AE")],["raw","iso2"])
 lk=F.broadcast(names.unionByName(iso).unionByName(aliases).dropDuplicates(["raw"]))

 s=lk.alias("sl"); q=lk.alias("ql")
 xn=(x.join(s,F.upper(F.trim(F.col("Sender_bank_location")))==F.col("sl.raw"),"left")
       .join(q,F.upper(F.trim(F.col("Receiver_bank_location")))==F.col("ql.raw"),"left")
       .withColumn("sender_iso",F.col("sl.iso2"))
       .withColumn("receiver_iso",F.col("ql.iso2")))

 gs=F.broadcast(g.select(F.upper(F.trim("iso2")).alias("siso"),
                         F.col("risk_tier").alias("stier"),
                         F.col("is_sanctioned").cast("int").alias("ssan")))
 gq=F.broadcast(g.select(F.upper(F.trim("iso2")).alias("qiso"),
                         F.col("risk_tier").alias("qtier"),
                         F.col("is_sanctioned").cast("int").alias("qsan")))

 z=(xn.join(gs,F.col("sender_iso")==F.col("siso"),"left")
      .join(gq,F.col("receiver_iso")==F.col("qiso"),"left"))

 out=(z.select(
   "development_row_id",
   ((F.col("stier")=="HIGH")|(F.col("qtier")=="HIGH")).cast("int")
      .alias("SCN_HIGH_RISK_GEOGRAPHY"),
   ((F.coalesce(F.col("ssan"),F.lit(0))==1)|
    (F.coalesce(F.col("qsan"),F.lit(0))==1)).cast("int")
      .alias("SCN_SANCTIONED_GEOGRAPHY"))
   .orderBy("development_row_id").cache())

 if out.count()!=n:
  raise RuntimeError("Governed geography output row count changed during joins.")

 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 out.coalesce(1).write.mode("overwrite").option("header",True).csv(a.out)

 print("\n=== GOVERNED GEOGRAPHY SCENARIOS — DEVELOPMENT ===")
 print(f"Rows: {n:,}")
 print("High-risk geography:",out.filter("SCN_HIGH_RISK_GEOGRAPHY=1").count())
 print("Sanctioned geography:",out.filter("SCN_SANCTIONED_GEOGRAPHY=1").count())
 print("Stable development_row_id uses zipWithIndex and aligns with Pandas candidate/frozen artifacts.")
 print("Policy-driven definitions. No HOLDOUT accessed.")
 spark.stop()

if __name__=="__main__":
 main()
