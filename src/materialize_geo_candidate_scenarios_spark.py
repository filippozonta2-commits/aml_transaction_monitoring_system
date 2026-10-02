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
 # Basel-style index: larger Overall Score means higher AML/CFT risk.
 risk_expr=F.col(risk).cast("double")
 rr=r.select(F.upper(F.trim(F.col(country).cast("string"))).alias("country_key"),
             risk_expr.alias("risk_score"),
             *(([F.col(sanction).alias("sanctioned")]) if sanction else [])).dropDuplicates(["country_key"])
 risk_cut=rr.approxQuantile("risk_score",[0.90],0.001)[0]
 print(f"Country-risk high-risk candidate cutoff (reference 90th percentile): {risk_cut}",flush=True)
 # Transaction locations are mostly country names plus UK/USA/UAE aliases, while
 # the reference key is ISO-2. Build a governed normalization from the reference itself.
 names=r.select(F.upper(F.trim(F.col("Country").cast("string"))).alias("raw_key"),
                F.upper(F.trim(F.col(country).cast("string"))).alias("country_key"))
 aliases=spark.createDataFrame([("UK","GB"),("USA","US"),("UAE","AE")],["raw_key","country_key"])
 lookup=names.unionByName(aliases).dropDuplicates(["raw_key"])
 sx=F.broadcast(lookup.alias("sl")); qx=F.broadcast(lookup.alias("ql"))
 xn=(x.join(sx,F.upper(F.trim(F.col("Sender_bank_location")))==F.col("sl.raw_key"),"left")
       .join(qx,F.upper(F.trim(F.col("Receiver_bank_location")))==F.col("ql.raw_key"),"left")
       .withColumn("sender_iso",F.coalesce(F.col("sl.country_key"),F.upper(F.trim(F.col("Sender_bank_location")))))
       .withColumn("receiver_iso",F.coalesce(F.col("ql.country_key"),F.upper(F.trim(F.col("Receiver_bank_location"))))))
 matched=xn.select("development_row_id","sender_iso","receiver_iso")
 s=F.broadcast(rr.alias("s")); q=F.broadcast(rr.alias("q"))
 z=(xn.join(s,F.col("sender_iso")==F.col("s.country_key"),"left")
      .join(q,F.col("receiver_iso")==F.col("q.country_key"),"left"))
 # Diagnostics: distinguish normalization failure from genuinely no high-risk exposure.
 cov=z.select(
   F.avg(F.col("s.risk_score").isNotNull().cast("double")).alias("sender_risk_coverage"),
   F.avg(F.col("q.risk_score").isNotNull().cast("double")).alias("receiver_risk_coverage"),
   F.max("s.risk_score").alias("sender_max_risk"),F.max("q.risk_score").alias("receiver_max_risk")
 ).first()
 print(f"Risk-score coverage sender={cov.sender_risk_coverage:.2%} receiver={cov.receiver_risk_coverage:.2%}",flush=True)
 print(f"Max transaction-country risk sender={cov.sender_max_risk} receiver={cov.receiver_max_risk}",flush=True)
 high=((F.col("s.risk_score")>=F.lit(risk_cut))|(F.col("q.risk_score")>=F.lit(risk_cut)))
 if sanction:
  def truth(c):
   # The reference stores descriptive sanctions text, not a Boolean flag.
   v=F.lower(F.trim(c.cast("string")))
   return c.isNotNull() & (F.length(v)>0)
  sanctioned=(truth(F.col("s.sanctioned"))|truth(F.col("q.sanctioned")))
  out=z.select(F.col("development_row_id"),high.cast("int").alias("SCN_HIGH_RISK_GEOGRAPHY"),
               sanctioned.cast("int").alias("SCN_SANCTIONED_GEOGRAPHY"))
 else:
  out=z.select(F.col("development_row_id"),high.cast("int").alias("SCN_HIGH_RISK_GEOGRAPHY"),
               F.lit(0).cast("int").alias("SCN_SANCTIONED_GEOGRAPHY"))
 out.write.mode("overwrite").option("header",True).csv(str(Path(a.out)/"geo_flags"))
 print("High-risk geography triggers:",out.filter("SCN_HIGH_RISK_GEOGRAPHY=1").count())
 print("Sanctioned geography triggers:",out.filter("SCN_SANCTIONED_GEOGRAPHY=1").count())
 if not sanction: print("No explicit sanctions field found; sanctioned flag intentionally remains zero.")
 print("No HOLDOUT accessed."); spark.stop()
if __name__=="__main__": main()
