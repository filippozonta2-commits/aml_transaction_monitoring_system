"""Evaluate shortlisted cross-border currency-mismatch variants incrementally
against the exact six frozen DEVELOPMENT scenarios.

DEVELOPMENT only. Diagnostic threshold comparison; never accesses HOLDOUT and
does not freeze a candidate.
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
    ap.add_argument("--out",default="results/candidate_scenarios/cross_border_mismatch_incremental.csv")
    a=ap.parse_args()

    # Exact frozen transaction-level baseline.
    fr=pd.read_csv(a.frozen_non_ds)
    ds=read_spark_csv(a.frozen_ds)
    if ds is None:
        raise FileNotFoundError("Missing frozen Deposit-Send artifact.")
    frozen=["SCN_STRUCTURING","SCN_DEPOSIT_SEND","SCN_FAN_OUT","SCN_FAN_IN","SCN_CASH_WITHDRAWAL","SCN_SMURFING"]
    fr=fr[["development_row_id"]+[c for c in frozen if c!="SCN_DEPOSIT_SEND"]]
    fr=fr.merge(ds[["development_row_id","SCN_DEPOSIT_SEND"]],on="development_row_id",validate="one_to_one")
    fr=fr.sort_values("development_row_id").reset_index(drop=True)
    fr["frozen_alert"]=fr[frozen].eq(1).any(axis=1).astype("int8")

    spark=(SparkSession.builder.appName("aml-mismatch-incremental")
           .master("local[*]").config("spark.sql.shuffle.partitions","16").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    x=(spark.read.option("header",True).option("inferSchema",True).csv(a.development)
       .withColumn("Amount",F.col("Amount").cast("double")))
    # Stable row id aligned to Pandas DEVELOPMENT artifacts.
    schema=x.schema
    x=spark.createDataFrame(x.rdd.zipWithIndex().map(lambda ri: tuple(ri[0])+ (ri[1],)),
                            schema.add("development_row_id","long"))
    pt=F.lower(F.trim(F.col("Payment_type")))
    applicable=~pt.isin("cash withdrawal","cash deposit")
    base=(applicable &
          (F.upper(F.trim("Sender_bank_location"))!=F.upper(F.trim("Receiver_bank_location"))) &
          (F.upper(F.trim("Payment_currency"))!=F.upper(F.trim("Received_currency"))))

    # DEVELOPMENT-wide absolute candidates.
    qvals=x.approxQuantile("Amount",[.95,.975],.001)
    flags=x.select("development_row_id","Is_laundering","Payment_type","Sender_bank_location","Receiver_bank_location","Payment_currency","Received_currency","Amount")
    for q,v in zip([.95,.975],qvals):
        flags=flags.withColumn(f"MISMATCH_Q{str(q).replace('.','_')}",
            (base & (F.col("Amount")>=F.lit(v))).cast("int"))

    # Prior sender history only: no future-row leakage.
    ts=F.to_timestamp(F.concat_ws(" ",F.col("Date"),F.col("Time")))
    w=(Window.partitionBy("Sender_account").orderBy(F.col("_ts").cast("long"))
       .rowsBetween(Window.unboundedPreceding,-1))
    z=(x.withColumn("_ts",ts)
       .withColumn("_hist_avg",F.avg("Amount").over(w))
       .withColumn("_hist_sd",F.stddev_samp("Amount").over(w))
       .withColumn("_hist_n",F.count("Amount").over(w)))
    for k in [3,4]:
        f=(base & (F.col("_hist_n")>=10) & (F.col("_hist_sd")>0) &
           (F.col("Amount")>=F.col("_hist_avg")+F.lit(float(k))*F.col("_hist_sd"))).cast("int")
        zz=z.select("development_row_id",f.alias(f"MISMATCH_Z{k}"))
        flags=flags.join(zz,"development_row_id","inner")

    pdf=flags.orderBy("development_row_id").toPandas()
    if len(pdf)!=len(fr) or not np.array_equal(pdf.development_row_id.to_numpy(),fr.development_row_id.to_numpy()):
        raise ValueError("Frozen/mismatch DEVELOPMENT row ids do not align.")
    y=pdf.Is_laundering.eq(1); baseline=fr.frozen_alert.eq(1)
    rows=[]
    for c in ["MISMATCH_Q0_95","MISMATCH_Q0_975","MISMATCH_Z3","MISMATCH_Z4"]:
        m=pdf[c].eq(1); inc=m&~baseline; ih=inc&y
        n=int(m.sum()); h=int((m&y).sum()); ni=int(inc.sum()); hi=int(ih.sum())
        rows.append(dict(candidate=c,alerts=n,aml_hits=h,precision=h/n if n else 0,
                         overlap_frozen=int((m&baseline).sum()),
                         new_alerts=ni,new_aml_hits=hi,
                         marginal_precision=hi/ni if ni else 0,
                         incremental_recall=hi/int(y.sum()) if y.sum() else 0))
    out=pd.DataFrame(rows).sort_values(["new_aml_hits","marginal_precision"],ascending=False)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
    print("\n=== CROSS-BORDER MISMATCH — INCREMENTAL VS 6 FROZEN SCENARIOS ===")
    print(f"Frozen baseline: alerts={int(baseline.sum()):,} | AML={int((baseline&y).sum()):,}/{int(y.sum()):,}")
    print(out.to_string(index=False))
    print("\nSemantic cash guard active. DEVELOPMENT only. No threshold frozen. No HOLDOUT accessed.")
    print("Saved:",a.out)
    spark.stop()

if __name__=="__main__":
    main()
