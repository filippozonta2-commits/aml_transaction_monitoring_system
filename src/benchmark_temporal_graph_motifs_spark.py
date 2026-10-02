"""Benchmark 48h/72h temporal graph motifs with PySpark on DEVELOPMENT only."""
import argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

def motifs(x,h):
 a=x.alias("a"); b=x.alias("b"); sec=F.lit(h*3600)
 p=(a.join(b,(F.col("a.dst")==F.col("b.src"))&(F.col("b.epoch")>F.col("a.epoch"))&((F.col("b.epoch")-F.col("a.epoch"))<=sec))
    .filter((F.col("a.src")!=F.col("b.dst"))&(F.col("a.src")!=F.col("a.dst"))&(F.col("b.src")!=F.col("b.dst")))
    .select(F.col("a.src").alias("a"),F.col("a.dst").alias("b"),F.col("b.dst").alias("c"),
            F.col("a.epoch").alias("t1"),F.col("b.epoch").alias("t2")).dropDuplicates()).cache()
 sg=p.groupBy("a","c").agg(F.countDistinct("b").alias("branches")).filter("branches>=3")
 lfo=p.groupBy("a").agg(F.countDistinct("b").alias("first_hop"),F.countDistinct("c").alias("second_hop")).filter("first_hop>=3 AND second_hop>=3")
 lfi=p.groupBy("c").agg(F.countDistinct("b").alias("first_hop"),F.countDistinct("a").alias("second_hop")).filter("first_hop>=3 AND second_hop>=3")
 e=x.select(F.col("src").alias("csrc"),F.col("dst").alias("cdst"),F.col("epoch").alias("t3"))
 cyc=(p.join(e,(F.col("c")==F.col("csrc"))&(F.col("a")==F.col("cdst"))&(F.col("t3")>F.col("t2"))&((F.col("t3")-F.col("t1"))<=sec))
      .select("a","b","c","t1","t2","t3").dropDuplicates())
 counts={"scatter_gather":sg.count(),"layered_fan_out":lfo.count(),"layered_fan_in":lfi.count(),"circular_movement":cyc.count()}
 p.unpersist(); return counts

def main():
 p=argparse.ArgumentParser(); p.add_argument("--development",default="data/temporal/SAML-D_development.csv"); p.add_argument("--out",default="results/candidate_scenarios/temporal_graph_benchmark"); a=p.parse_args()
 spark=(SparkSession.builder.appName("aml-temporal-graph-benchmark").config("spark.sql.shuffle.partitions","48").getOrCreate()); spark.sparkContext.setLogLevel("WARN")
 raw=spark.read.option("header",True).option("inferSchema",True).csv(a.development)
 x=(raw.withColumn("ts",F.to_timestamp(F.concat_ws(" ",F.col("Date").cast("string"),F.col("Time").cast("string"))))
    .filter(F.col("ts").isNotNull()).select(F.col("Sender_account").cast("string").alias("src"),F.col("Receiver_account").cast("string").alias("dst"),F.col("ts").cast("long").alias("epoch"))
    .filter(F.col("src")!=F.col("dst")).dropDuplicates().repartition(48,"src").cache())
 rows=[]
 for h in (48,72):
  print(f"Building temporal motifs for {h}h...",flush=True); c=motifs(x,h)
  for k,v in c.items(): rows.append((h,k,v)); print(f"{h}h {k}: {v:,}",flush=True)
 spark.createDataFrame(rows,["window_hours","scenario","motifs"]).coalesce(1).write.mode("overwrite").option("header",True).csv(a.out)
 print("No HOLDOUT accessed."); spark.stop()
if __name__=="__main__": main()
