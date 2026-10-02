"""Fast PySpark evaluation of selected Deposit-Send V4 receiver-flow candidates.
TRAIN supplies warm-up context; metrics are DEVELOPMENT-only. HOLDOUT is never accessed.
"""
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

TRAIN="data/temporal/SAML-D_train.csv"; DEV="data/temporal/SAML-D_development.csv"
CANDS=[("A_recall",72,.10,5.0,False),("B_balanced",72,.50,3.0,False),
       ("C_tighter",72,.50,2.0,False),("D_cb_baseline",72,.50,3.0,True),
       ("E_48h",48,.25,3.0,False)]

def load(spark,path,split):
    # Keep Date/Time as strings. Spark schema inference can misparse SAML-D dates.
    d=(spark.read.option("header",True).option("inferSchema",False).csv(path)
       .select("Time","Date","Sender_account","Receiver_account","Amount","Payment_type","Is_laundering","Laundering_type"))
    date_raw=F.trim(F.col("Date"))
    time_raw=F.trim(F.col("Time"))
    date_parsed=F.coalesce(
        F.to_date(date_raw,"yyyy-MM-dd"),
        F.to_date(date_raw,"M/d/yyyy"),
        F.to_date(date_raw,"MM/dd/yyyy"),
        F.to_date(date_raw,"yyyy/MM/dd")
    )
    ts=F.to_timestamp(F.concat_ws(" ",F.date_format(date_parsed,"yyyy-MM-dd"),time_raw))
    return (d.withColumn("ts",ts)
             .withColumn("Sender_account",F.col("Sender_account").cast("long"))
             .withColumn("Receiver_account",F.col("Receiver_account").cast("long"))
             .withColumn("Amount",F.col("Amount").cast("double"))
             .withColumn("Is_laundering",F.col("Is_laundering").cast("int"))
             .withColumn("split",F.lit(split))
             .filter(F.col("ts").isNotNull()))

def main():
    spark=(SparkSession.builder.appName("DepositSendV4Evaluation")
           .master("local[*]")
           .config("spark.sql.shuffle.partitions","24")
           .config("spark.driver.memory","4g")
           .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    print("Loading TRAIN warm-up + DEVELOPMENT with Spark...")
    tr=load(spark,TRAIN,"train"); dev=load(spark,DEV,"development").cache()
    n=dev.count()
    null_ts=dev.filter(F.col("ts").isNull()).count()
    print(f"Development rows parsed: {n:,} | NULL timestamps after filter: {null_ts:,}")
    if n==0:
        raise RuntimeError("No development rows were parsed. Check Date/Time formats.")
    aml_n=dev.filter(F.col("Is_laundering")==1).count(); overall=aml_n/n
    start=dev.agg(F.min("ts")).first()[0]
    print(f"Development start: {start}")
    print("HOLDOUT is not accessed.")

    ev=tr.filter(F.col("ts")>=F.lit(start)-F.expr("INTERVAL 72 HOURS")).unionByName(dev).cache()
    anchors=(dev.withColumn("anchor_id",F.monotonically_increasing_id())
             .select("anchor_id",F.col("ts").alias("anchor_ts"),F.col("Receiver_account").alias("focal"),
                     F.col("Amount").alias("anchor_amount"),"Is_laundering","Laundering_type")).cache()
    outs=ev.select(F.col("Sender_account").alias("focal"),F.col("ts").alias("out_ts"),
                   F.col("Amount").alias("out_amount"),F.col("Payment_type").alias("out_payment"))
    j=(anchors.alias("a").join(outs.alias("o"),
       (F.col("a.focal")==F.col("o.focal"))&
       (F.col("o.out_ts")>F.col("a.anchor_ts"))&
       (F.col("o.out_ts")<=F.col("a.anchor_ts")+F.expr("INTERVAL 72 HOURS")),"left")
       .withColumn("hours_after",(F.unix_timestamp("out_ts")-F.unix_timestamp("anchor_ts"))/3600.0))
    feat=(j.groupBy("anchor_id","anchor_amount","Is_laundering","Laundering_type")
          .agg(F.sum(F.when(F.col("hours_after")<=48,F.col("out_amount")).otherwise(0.0)).alias("sum48"),
               F.sum(F.col("out_amount")).alias("sum72"),
               F.max(F.when((F.col("hours_after")<=48)&(F.col("out_payment")=="Cross-border"),1).otherwise(0)).alias("cb48"),
               F.max(F.when(F.col("out_payment")=="Cross-border",1).otherwise(0)).alias("cb72"))
          .withColumn("ratio48",F.when(F.col("anchor_amount")!=0,F.col("sum48")/F.col("anchor_amount")))
          .withColumn("ratio72",F.when(F.col("anchor_amount")!=0,F.col("sum72")/F.col("anchor_amount")))
          .cache())
    feat.count()

    target_n=anchors.filter((F.col("Is_laundering")==1)&(F.col("Laundering_type")=="Deposit-Send")).count()
    rows=[]
    for name,h,lo,hi,reqcb in CANDS:
        print(f"Evaluating {name}...")
        ratio=F.col("ratio48" if h==48 else "ratio72"); cb=F.col("cb48" if h==48 else "cb72")
        cond=ratio.between(lo,hi)
        if reqcb: cond=cond&(cb==1)
        vals=(feat.filter(cond)
              .agg(F.count("*").alias("trig"),F.sum("Is_laundering").alias("aml"),
                   F.sum(F.when((F.col("Is_laundering")==1)&(F.col("Laundering_type")=="Deposit-Send"),1).otherwise(0)).alias("ds"))
              .first())
        trig=int(vals.trig or 0); aml=int(vals.aml or 0); ds=int(vals.ds or 0)
        p=aml/trig if trig else 0
        rows.append((name,h,lo,hi,reqcb,trig,trig/n,aml,p,p/overall if overall else 0,ds,ds/target_n if target_n else 0))

    cols=["Candidate","Horizon_hours","Min_ratio","Max_ratio","Require_cross_border",
          "Triggered_transactions","Trigger_rate","AML_cases","Precision","Lift_vs_overall_AML_rate",
          "Deposit_Send_hits","Deposit_Send_recall"]
    res=spark.createDataFrame(rows,cols).orderBy(F.desc("Deposit_Send_recall"),F.desc("Precision"))
    print("\n=== DEPOSIT-SEND V4 CANDIDATE EVALUATION ===")
    res.show(truncate=False)
    out="results/deposit_send_v4_evaluation"; Path(out).mkdir(parents=True,exist_ok=True)
    res.coalesce(1).write.mode("overwrite").option("header",True).csv(out+"/spark_candidate_evaluation")
    print(f"Overall development AML rate: {overall:.4%} | Deposit-Send targets: {target_n}")
    print("Development-only. HOLDOUT was not accessed.")
    spark.stop()

if __name__=="__main__": main()
