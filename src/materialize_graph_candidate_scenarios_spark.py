"""PySpark materializer for second-order AML network candidates.

DEVELOPMENT ONLY. No HOLDOUT access. Candidate engineering pass; no thresholds
are frozen here. Uses bounded day-level graph motifs to keep joins tractable.
"""
from pathlib import Path
import argparse
from pyspark.sql import SparkSession, functions as F

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--development",default="data/temporal/SAML-D_development.csv")
 p.add_argument("--out",default="results/candidate_scenarios/graph_spark")
 a=p.parse_args()
 spark=(SparkSession.builder.appName("aml-graph-candidate-scenarios")
        .config("spark.sql.shuffle.partitions","48").getOrCreate())
 spark.sparkContext.setLogLevel("WARN")
 print("Loading DEVELOPMENT with Spark...",flush=True)
 x=(spark.read.option("header",True).option("inferSchema",True).csv(a.development)
    .withColumn("development_row_id",F.monotonically_increasing_id())
    .withColumn("ts",F.to_timestamp(F.concat_ws(" ",F.col("Date").cast("string"),F.col("Time").cast("string"))))
    .filter(F.col("ts").isNotNull())
    .withColumn("day",F.to_date("ts"))
    .select("development_row_id","ts","day",
            F.col("Sender_account").cast("string").alias("src"),
            F.col("Receiver_account").cast("string").alias("dst"),
            F.col("Amount").cast("double").alias("amount")))
 # Aggregate duplicate edges first: motif discovery is about account relations.
 e=(x.groupBy("day","src","dst").agg(F.count("*").alias("edge_tx"),F.sum("amount").alias("edge_amount"))
      .filter(F.col("src")!=F.col("dst")).cache())
 outdeg=e.groupBy("day","src").agg(F.countDistinct("dst").alias("out_degree"))
 indeg=e.groupBy("day","dst").agg(F.countDistinct("src").alias("in_degree"))
 # Gather -> Scatter: intermediary has >=3 distinct inbound and >=3 outbound counterparties same day.
 xi=x.alias("x"); ii=indeg.alias("ii"); oo=outdeg.alias("oo")
 gs=(xi.join(ii,(F.col("x.day")==F.col("ii.day"))&(F.col("x.src")==F.col("ii.dst")),"left")
       .join(oo,(F.col("x.day")==F.col("oo.day"))&(F.col("x.src")==F.col("oo.src")),"left")
       .select(F.col("x.development_row_id"),
               F.coalesce(F.col("ii.in_degree"),F.lit(0)).alias("in_degree"),
               F.coalesce(F.col("oo.out_degree"),F.lit(0)).alias("out_degree"))
       .withColumn("SCN_GATHER_SCATTER_SPARK",((F.col("in_degree")>=3)&(F.col("out_degree")>=3)).cast("int")))
 # Two-hop paths A->B->C on same day.
 a1=e.alias("a"); b1=e.alias("b")
 paths=(a1.join(b1,(F.col("a.day")==F.col("b.day"))&(F.col("a.dst")==F.col("b.src")))
        .filter((F.col("a.src")!=F.col("b.dst"))&(F.col("a.src")!=F.col("a.dst"))&(F.col("b.src")!=F.col("b.dst")))
        .select(F.col("a.day").alias("day"),F.col("a.src").alias("a"),F.col("a.dst").alias("b"),F.col("b.dst").alias("c")).dropDuplicates())
 # Scatter-Gather: same origin reaches same destination through >=3 distinct intermediaries.
 sg=(paths.groupBy("day","a","c").agg(F.countDistinct("b").alias("branches")).filter(F.col("branches")>=3)
     .select(F.col("day"),F.col("a").alias("src"),F.col("c").alias("dst"),"branches"))
 # Layered fan-out: origin has >=3 first-hop branches whose intermediaries collectively reach >=3 second-hop accounts.
 lfo=(paths.groupBy("day","a").agg(F.countDistinct("b").alias("first_hop"),F.countDistinct("c").alias("second_hop"))
      .filter((F.col("first_hop")>=3)&(F.col("second_hop")>=3))
      .select(F.col("day"),F.col("a").alias("account"),"first_hop","second_hop"))
 # Layered fan-in: reverse motif, >=3 upstream origins through >=3 intermediaries into destination.
 lfi=(paths.groupBy("day","c").agg(F.countDistinct("b").alias("first_hop"),F.countDistinct("a").alias("second_hop"))
      .filter((F.col("first_hop")>=3)&(F.col("second_hop")>=3))
      .select(F.col("day"),F.col("c").alias("account"),"first_hop","second_hop"))
 # Directed 3-cycle A->B->C->A, canonicalized to avoid counting rotations as separate motifs.
 c1=e.alias("c1")
 cyc=(paths.alias("p").join(c1,(F.col("p.day")==F.col("c1.day"))&(F.col("p.c")==F.col("c1.src"))&(F.col("p.a")==F.col("c1.dst")))
      .select(F.col("p.day").alias("day"),F.col("p.a").alias("a"),F.col("p.b").alias("b"),F.col("p.c").alias("c"))
      .filter((F.col("a")!=F.col("b"))&(F.col("b")!=F.col("c"))&(F.col("a")!=F.col("c")))
      .withColumn("cycle_key",F.concat_ws("|",F.array_sort(F.array("a","b","c"))))
      .dropDuplicates(["day","cycle_key"]))
 for name,df in [("scatter_gather",sg),("layered_fan_out",lfo),("layered_fan_in",lfi),("circular_movement",cyc)]:
  path=str(Path(a.out)/name); df.write.mode("overwrite").option("header",True).csv(path)
  print(f"{name}: {df.count():,} motifs",flush=True)
 gs.select("development_row_id","SCN_GATHER_SCATTER_SPARK").write.mode("overwrite").option("header",True).csv(str(Path(a.out)/"gather_scatter_flags"))
 print("Graph candidate artifacts saved:",a.out)
 print("Candidate motif thresholds only. No HOLDOUT accessed.")
 spark.stop()
if __name__=="__main__": main()
