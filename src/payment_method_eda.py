"""Payment-method segmentation EDA for SAML-D development data.

Tests whether transaction channels have materially different AML prevalence,
typology mix, amounts, and geography/FX behavior. This is diagnostic only:
Payment_type is not assumed to be suspicious a priori.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,
                   default=Path("data/temporal/SAML-D_development.csv"))
    p.add_argument("--output-dir",type=Path,
                   default=Path("results/payment_method_eda"))
    a=p.parse_args()

    usecols=[
        "Amount","Payment_type","Is_laundering","Laundering_type",
        "Payment_currency","Received_currency",
        "Sender_bank_location","Receiver_bank_location",
    ]
    print("Loading DEVELOPMENT only...")
    df=pd.read_csv(a.input,usecols=usecols)
    df["Cross_border"]=(df["Sender_bank_location"]!=df["Receiver_bank_location"]).astype("int8")
    df["Currency_mismatch"]=(df["Payment_currency"]!=df["Received_currency"]).astype("int8")

    overall_aml=df["Is_laundering"].mean()

    channel=df.groupby("Payment_type",dropna=False).agg(
        Transactions=("Is_laundering","size"),
        AML_transactions=("Is_laundering","sum"),
        AML_rate=("Is_laundering","mean"),
        Amount_median=("Amount","median"),
        Amount_mean=("Amount","mean"),
        Amount_p95=("Amount",lambda x:x.quantile(.95)),
        Cross_border_rate=("Cross_border","mean"),
        Currency_mismatch_rate=("Currency_mismatch","mean"),
    ).reset_index()
    channel["Share_of_transactions"]=channel["Transactions"]/len(df)
    channel["AML_rate_vs_overall"]=channel["AML_rate"]/overall_aml
    channel=channel.sort_values("Transactions",ascending=False)

    aml=df[df["Is_laundering"].eq(1)]
    typ=pd.crosstab(
        aml["Payment_type"],aml["Laundering_type"],margins=False
    )
    typ_share=typ.div(typ.sum(axis=1),axis=0)

    long=(aml.groupby(["Payment_type","Laundering_type"],dropna=False)
          .agg(AML_transactions=("Is_laundering","size"),
               Amount_median=("Amount","median"),
               Amount_p95=("Amount",lambda x:x.quantile(.95)),
               Cross_border_rate=("Cross_border","mean"),
               Currency_mismatch_rate=("Currency_mismatch","mean"))
          .reset_index())
    totals=long.groupby("Payment_type")["AML_transactions"].transform("sum")
    long["Share_within_payment_method"]=long["AML_transactions"]/totals
    long=long.sort_values(["Payment_type","AML_transactions"],
                          ascending=[True,False])

    a.output_dir.mkdir(parents=True,exist_ok=True)
    channel.to_csv(a.output_dir/"payment_method_summary.csv",index=False)
    typ.to_csv(a.output_dir/"payment_method_typology_counts.csv")
    typ_share.to_csv(a.output_dir/"payment_method_typology_shares.csv")
    long.to_csv(a.output_dir/"payment_method_typology_detail.csv",index=False)

    print("\n=== PAYMENT METHOD SUMMARY ===")
    print(channel.to_string(index=False))
    print("\n=== TOP AML TYPOLOGIES BY PAYMENT METHOD ===")
    for method,g in long.groupby("Payment_type",dropna=False):
        print(f"\n[{method}]")
        print(g.head(6)[[
            "Laundering_type","AML_transactions",
            "Share_within_payment_method","Amount_median",
            "Cross_border_rate","Currency_mismatch_rate"
        ]].to_string(index=False))
    print(f"\nOverall development AML rate: {overall_aml:.4%}")
    print(f"Outputs saved to: {a.output_dir}")


if __name__=="__main__":
    main()
