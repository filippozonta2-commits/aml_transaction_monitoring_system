"""Apply the governed geography registry to HOLDOUT without tuning."""
import argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
def main():
 p=argparse.ArgumentParser(); p.add_argument("--holdout",default="data/temporal/SAML-D_holdout.csv");p.add_argument("--governance",default="results/governance/country_risk");p.add_argument("--out",default="results/holdout_v3/governed_geo");a=p.parse_args()
 spark=(SparkSession.builder.appName("AML-V3-Holdout-Geo").master("local[*]").config("spark.sql.shuffle.partitions","16").getOrCreate());spark.sparkContext.setLogLevel("WARN")
 x=spark.read.option("header",True).option("inferSchema",True).csv(a.holdout);schema=x.schema.add("holdout_row_id","long",False)
 x=spark.createDataFrame(x.rdd.zipWithIndex().map(lambda z:tuple(z[0])+(z[1],)),schema).cache();n=x.count()
 g=spark.read.option("header",True).option("inferSchema",True).csv(a.governance)
 names=g.select(F.upper(F.trim("country")).alias("raw"),F.upper(F.trim("iso2")).alias("iso2"));iso=g.select(F.upper(F.trim("iso2")).alias("raw"),F.upper(F.trim("iso2")).alias("iso2"))
 aliases=spark.createDataFrame([("UK","GB"),("USA","US"),("UAE","AE")],["raw","iso2"]);lk=F.broadcast(names.unionByName(iso).unionByName(aliases).dropDuplicates(["raw"]))
 xn=(x.join(lk.alias("sl"),F.upper(F.trim(F.col("Sender_bank_location")))==F.col("sl.raw"),"left").join(lk.alias("ql"),F.upper(F.trim(F.col("Receiver_bank_location")))==F.col("ql.raw"),"left")
     .withColumn("sender_iso",F.col("sl.iso2")).withColumn("receiver_iso",F.col("ql.iso2")))
 gs=F.broadcast(g.select(F.upper(F.trim("iso2")).alias("siso"),F.col("risk_tier").alias("stier"),F.col("is_sanctioned").cast("int").alias("ssan")))
 gq=F.broadcast(g.select(F.upper(F.trim("iso2")).alias("qiso"),F.col("risk_tier").alias("qtier"),F.col("is_sanctioned").cast("int").alias("qsan")))
 z=xn.join(gs,F.col("sender_iso")==F.col("siso"),"left").join(gq,F.col("receiver_iso")==F.col("qiso"),"left")
 out=z.select("holdout_row_id",((F.col("stier")=="HIGH")|(F.col("qtier")=="HIGH")).cast("int").alias("SCN_HIGH_RISK_GEOGRAPHY"),((F.coalesce(F.col("ssan"),F.lit(0))==1)|(F.coalesce(F.col("qsan"),F.lit(0))==1)).cast("int").alias("SCN_SANCTIONED_GEOGRAPHY")).orderBy("holdout_row_id").cache()
 if out.count()!=n: raise RuntimeError("Geography join changed row count")
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);out.coalesce(1).write.mode("overwrite").option("header",True).csv(a.out)
 print("=== GOVERNED GEOGRAPHY — HOLDOUT ===");print("HIGH:",out.filter("SCN_HIGH_RISK_GEOGRAPHY=1").count());print("SANCTIONED:",out.filter("SCN_SANCTIONED_GEOGRAPHY=1").count());print("Saved:",a.out);spark.stop()
if __name__=="__main__":main()
