"""PySpark geography candidate materializer — DEVELOPMENT only.

Uses the project's governed country-risk reference. It never invents a sanctions
list: sanctioned geography is emitted only if the reference contains an
explicit sanctions/restricted indicator.
"""
from pathlib import Path
import argparse
from pyspark.sql import SparkSession, functions as F

def pick(cols,candidates):
 low={c.lower():c for c in cols}
 for x in candidates:
  if x.lower() in low:return low[x.lower()]
 return None

def main():
 p=argparse.ArgumentParser(); p.add_argument("--development",default="data/temporal/SAML-D_development.csv")
 p.add_argument("--country-risk",default="data/country_risk.csv"); p.add_argument("--out",default="results/candidate_scenarios/geo_spark"); a=p.parse_args()
 spark=SparkSession.builder.appName("aml-geo-candidate-scenarios").config("spark.sql.shuffle.partitions","32").getOrCreate(); spark.sparkContext.setLogLevel("WARN")
 x=(spark.read.option("header",True).option("inferSchema",True).csv(a.development)
    .withColumn("development_row_id",F.monotonically_increasing_id()))
 r=spark.read.option("header",True).option("inferSchema",True).csv(a.country_risk)
 country=pick(r.columns,["ISO Code","country_iso2","iso2","country_code","code","country"])
 risk=pick(r.columns,["Overall Score","ML/TF Risk","risk_level","risk_rating","risk","country_risk"])
 sanction=pick(r.columns,["Sanction","sanctioned","sanctions","restricted","is_sanctioned"])
 if not country or not risk: raise ValueError(f"Country-risk schema unsupported: {r.columns}")
 risk_expr=F.col(risk).cast("double")
 rr=r.select(F.upper(F.trim(F.col(country).cast("string"))).alias("country_key"),
             risk_expr.alias("risk_score"),
             *(([F.col(sanction).alias("sanctioned")]) if sanction else [])).dropDuplicates(["country_key"])
 risk_cut=rr.approxQuantile("risk_score",[0.90],0.001)[0]
 print(f"Country-risk high-risk candidate cutoff (reference 90th percentile): {risk_cut}",flush=True)
 s=rr.alias("s"); q=rr.alias("q")
 z=(x.join(s,F.upper(F.trim(F.col("Sender_bank_location")))==F.col("s.country_key"),"left")
      .join(q,F.upper(F.trim(F.col("Receiver_bank_location")))==F.col("q.country_key"),"left"))
 high=((F.col("s.risk_score")>=F.lit(risk_cut))|(F.col("q.risk_score")>=F.lit(risk_cut)))
 out=z.select("development_row_id",high.cast("int").alias("SCN_HIGH_RISK_GEOGRAPHY"))
 if sanction:
  def truth(c): return F.lower(F.trim(c.cast("string"))).isin("1","true","yes","y","sanctioned","restricted")
  out=out.withColumn("SCN_SANCTIONED_GEOGRAPHY",(truth(F.col("s.sanctioned"))|truth(F.col("q.sanctioned"))).cast("int"))
 else:
  out=out.withColumn("SCN_SANCTIONED_GEOGRAPHY",F.lit(0).cast("int"))
 out.write.mode("overwrite").option("header",True).csv(str(Path(a.out)/"geo_flags"))
 print("High-risk geography triggers:",out.filter("SCN_HIGH_RISK_GEOGRAPHY=1").count())
 print("Sanctioned geography triggers:",out.filter("SCN_SANCTIONED_GEOGRAPHY=1").count())
 if not sanction: print("No explicit sanctions field found; sanctioned flag intentionally remains zero.")
 print("No HOLDOUT accessed."); spark.stop()
if __name__=="__main__": main()
