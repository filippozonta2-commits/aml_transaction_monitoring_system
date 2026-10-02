"""Benchmark unusual-amount candidates incrementally vs the six frozen scenarios.

DEVELOPMENT only. Compares the current rule with stricter account-relative
ratio rules and prior-sender-history z-score rules. No HOLDOUT access.
"""
from pathlib import Path
import argparse, pandas as pd, numpy as np
from pyspark.sql import SparkSession, functions as F, Window

def read_spark_csv(path):
    p=Path(path)
    files=list(p.glob("part-*.csv")) if p.exists() else []
    return pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--development",default="data/temporal/SAML-D_development.csv")
    ap.add_argument("--frozen-non-ds",default="results/unified_dashboard/materialized_non_deposit_send.csv")
    ap.add_argument("--frozen-ds",default="results/unified_dashboard/deposit_send_flags")
    ap.add_argument("--out",default="results/candidate_scenarios/unusual_amount_incremental.csv")
    a=ap.parse_args()

    fr=pd.read_csv(a.frozen_non_ds)
    ds=read_spark_csv(a.frozen_ds)
    if ds is None: raise FileNotFoundError("Missing frozen Deposit-Send artifact.")
    frozen=["SCN_STRUCTURING","SCN_DEPOSIT_SEND","SCN_FAN_OUT","SCN_FAN_IN","SCN_CASH_WITHDRAWAL","SCN_SMURFING"]
    fr=fr[["development_row_id"]+[c for c in frozen if c!="SCN_DEPOSIT_SEND"]]
    fr=fr.merge(ds[["development_row_id","SCN_DEPOSIT_SEND"]],on="development_row_id",validate="one_to_one")
    fr=fr.sort_values("development_row_id").reset_index(drop=True)
    fr["frozen_alert"]=fr[frozen].eq(1).any(axis=1).astype("int8")

    spark=(SparkSession.builder.appName("aml-unusual-amount-benchmark")
           .master("local[*]").config("spark.sql.shuffle.partitions","16").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    x=(spark.read.option("header",True).option("inferSchema",True).csv(a.development)
       .withColumn("Amount",F.col("Amount").cast("double")))
    schema=x.schema
    x=spark.createDataFrame(x.rdd.zipWithIndex().map(lambda ri: tuple(ri[0])+(ri[1],)),
                            schema.add("development_row_id","long"))

    q90=x.approxQuantile("Amount",[.90],.001)[0]
    acct=x.groupBy("Sender_account").agg(F.expr("percentile_approx(Amount, 0.5)").alias("acct_med"),
                                         F.count("*").alias("acct_n"))
    z=x.join(acct,"Sender_account","left")
    flags=x.select("development_row_id","Is_laundering")
    for ratio in [5,7.5,10]:
        name="UNUSUAL_RATIO_"+str(ratio).replace(".","_")
        f=((F.col("acct_n")>=5)&(F.col("acct_med")>0)&
           (F.col("Amount")/F.col("acct_med")>=F.lit(float(ratio)))&
           (F.col("Amount")>=F.lit(q90))).cast("int")
        flags=flags.join(z.select("development_row_id",f.alias(name)),"development_row_id","inner")

    ts=F.to_timestamp(F.concat_ws(" ",F.col("Date"),F.col("Time")))
    w=(Window.partitionBy("Sender_account").orderBy(F.col("_ts").cast("long"))
       .rowsBetween(Window.unboundedPreceding,-1))
    h=(x.withColumn("_ts",ts)
       .withColumn("_hist_avg",F.avg("Amount").over(w))
       .withColumn("_hist_sd",F.stddev_samp("Amount").over(w))
       .withColumn("_hist_n",F.count("Amount").over(w)))
    for k in [3,4,5]:
        name=f"UNUSUAL_PRIOR_Z{k}"
        f=((F.col("_hist_n")>=10)&(F.col("_hist_sd")>0)&
           (F.col("Amount")>=F.col("_hist_avg")+F.lit(float(k))*F.col("_hist_sd"))).cast("int")
        flags=flags.join(h.select("development_row_id",f.alias(name)),"development_row_id","inner")

    pdf=flags.orderBy("development_row_id").toPandas()
    if len(pdf)!=len(fr) or not np.array_equal(pdf.development_row_id.to_numpy(),fr.development_row_id.to_numpy()):
        raise ValueError("Frozen/unusual-amount DEVELOPMENT row ids do not align.")
    y=pdf.Is_laundering.eq(1); baseline=fr.frozen_alert.eq(1)
    rows=[]
    candidates=[c for c in pdf.columns if c.startswith("UNUSUAL_")]
    for c in candidates:
        m=pdf[c].eq(1); inc=m&~baseline
        n=int(m.sum()); h=int((m&y).sum()); ni=int(inc.sum()); hi=int((inc&y).sum())
        rows.append(dict(candidate=c,alerts=n,aml_hits=h,precision=h/n if n else 0,
                         overlap_frozen=int((m&baseline).sum()),new_alerts=ni,new_aml_hits=hi,
                         marginal_precision=hi/ni if ni else 0,
                         incremental_recall=hi/int(y.sum()) if y.sum() else 0))
    out=pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
    print()
    print("=== UNUSUAL AMOUNT BENCHMARK — INCREMENTAL VS 6 FROZEN SCENARIOS ===")
    print(f"Frozen baseline: alerts={int(baseline.sum()):,} | AML={int((baseline&y).sum()):,}/{int(y.sum()):,}")
    print(f"Development-wide q90 amount floor: {q90:,.2f}")
    print(out.to_string(index=False))
    print()
    print("Ratio variants reproduce/tighten the existing sender-median rule.")
    print("Prior-Z variants use only earlier sender transactions (>=10 history rows).")
    print("DEVELOPMENT only. No threshold frozen. No HOLDOUT accessed.")
    print("Saved:",a.out)
    spark.stop()

if __name__=="__main__":
    main()
