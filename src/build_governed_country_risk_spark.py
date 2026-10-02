"""Build an auditable AML country-risk registry from the project reference table.

Policy/reference layer only. It does not inspect DEVELOPMENT labels or HOLDOUT.
HIGH is an explicit governed score rule over the external country-risk reference,
not a percentile chosen from transaction/model performance. Sanctions stay separate.
"""
import argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

def nonempty(c):
 return c.isNotNull() & (F.length(F.trim(c.cast("string")))>0)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--country-risk",default="data/country_risk.csv")
 p.add_argument("--high-cutoff",type=float,default=6.5,
                help="Governed Overall Score cutoff for HIGH; policy parameter, not model-tuned.")
 p.add_argument("--elevated-cutoff",type=float,default=6.0,
                help="Governed Overall Score cutoff for ELEVATED.")
 p.add_argument("--out",default="results/governance/country_risk")
 a=p.parse_args()
 if a.elevated_cutoff>=a.high_cutoff:
  raise ValueError("elevated-cutoff must be lower than high-cutoff")
 spark=SparkSession.builder.appName("aml-governed-country-risk").getOrCreate()
 spark.sparkContext.setLogLevel("WARN")
 r=spark.read.option("header",True).option("inferSchema",True).csv(a.country_risk)
 score=F.col("Overall Score").cast("double")
 sanctioned=nonempty(F.col("Sanction"))
 tier=(F.when(score>=F.lit(a.high_cutoff),"HIGH")
         .when(score>=F.lit(a.elevated_cutoff),"ELEVATED")
         .otherwise("STANDARD"))
 rationale=(F.when(score>=F.lit(a.high_cutoff),
                    F.concat(F.lit("Overall Score >= "),F.lit(str(a.high_cutoff))))
              .when(score>=F.lit(a.elevated_cutoff),
                    F.concat(F.lit("Overall Score >= "),F.lit(str(a.elevated_cutoff))))
              .otherwise(F.lit("Below governed risk cutoffs")))
 out=(r.select(
   F.col("Country").alias("country"),
   F.upper(F.trim(F.col("ISO Code").cast("string"))).alias("iso2"),
   score.alias("overall_score"),
   F.col("ML/TF Risk").cast("string").alias("ml_tf_risk_source"),
   F.col("FATF").cast("string").alias("fatf_source"),
   F.col("Sanction").cast("string").alias("sanction_source"),
   tier.alias("risk_tier"),
   rationale.alias("risk_rationale"),
   F.lit("project_country_risk_reference").alias("risk_source"),
   sanctioned.cast("int").alias("is_sanctioned"))
   .filter(F.col("iso2").isNotNull()).dropDuplicates(["iso2"]).orderBy("iso2"))
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 out.coalesce(1).write.mode("overwrite").option("header",True).csv(a.out)
 print("\n=== GOVERNED COUNTRY-RISK REGISTRY ===")
 print(f"Policy cutoffs: HIGH >= {a.high_cutoff:.2f} | ELEVATED >= {a.elevated_cutoff:.2f}")
 out.groupBy("risk_tier").count().orderBy("risk_tier").show(truncate=False)
 print("Sanctioned jurisdictions:",out.filter("is_sanctioned=1").count())
 print("\nHIGH jurisdictions — governed reference rule:")
 out.filter("risk_tier='HIGH'").select("country","iso2","overall_score","risk_rationale").orderBy(F.desc("overall_score")).show(200,False)
 print("Saved:",a.out)
 print("Policy/reference artifact only. No DEVELOPMENT labels or HOLDOUT accessed.")
 spark.stop()
if __name__=="__main__": main()
