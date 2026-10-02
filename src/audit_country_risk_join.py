"""Audit and normalize SAML-D <-> country-risk country joins.

Read-only audit: writes reports under results/country_risk_audit and does not
modify raw data or access temporal HOLDOUT files.
"""
from pathlib import Path
import argparse
import pandas as pd
from country_normalization import country_key

def pick_col(df,candidates,label):
    lut={c.lower().strip():c for c in df.columns}
    for x in candidates:
        if x.lower() in lut:return lut[x.lower()]
    raise KeyError(f"Could not identify {label} column. Columns: {list(df.columns)}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--saml",type=Path,default=Path("data/SAML-D.csv"))
    p.add_argument("--risk",type=Path,default=Path("data/country_risk.csv"))
    p.add_argument("--out",type=Path,default=Path("results/country_risk_audit"))
    a=p.parse_args()
    print("Loading SAML-D country columns and country-risk table...")
    s=pd.read_csv(a.saml,usecols=lambda c:c in {"Sender_bank_location","Receiver_bank_location"})
    r=pd.read_csv(a.risk)
    rc=pick_col(r,["country","country_name","jurisdiction","name","Country"],"country-risk country")
    r["country_iso2_key"]=r[rc].map(country_key)
    dup=r[r["country_iso2_key"].notna() & r["country_iso2_key"].duplicated(False)]
    if len(dup):
        print(f"WARNING: {len(dup):,} country-risk rows share normalized keys; inspect duplicate report.")

    risk_keys=set(r["country_iso2_key"].dropna())
    rows=[]
    for side,col in [("sender","Sender_bank_location"),("receiver","Receiver_bank_location")]:
        vc=s[col].value_counts(dropna=False)
        for raw,n in vc.items():
            key=country_key(raw)
            rows.append({"side":side,"raw_country":raw,"country_iso2_key":key,
                         "transactions":int(n),"risk_match":key in risk_keys if key else False})
    rep=pd.DataFrame(rows)
    a.out.mkdir(parents=True,exist_ok=True)
    rep.to_csv(a.out/"country_join_coverage.csv",index=False)
    r.to_csv(a.out/"country_risk_normalized.csv",index=False)
    dup.to_csv(a.out/"country_risk_duplicate_keys.csv",index=False)

    print("\n=== COUNTRY NORMALIZATION / JOIN COVERAGE ===")
    for key,label in [("US","USA"),("GB","UK"),("AE","UAE")]:
        q=rep[rep["country_iso2_key"].eq(key)]
        print(f"{label:>4} -> {key}: SAML rows={int(q.transactions.sum()):,} | "
              f"raw variants={sorted(q.raw_country.astype(str).unique().tolist())} | "
              f"risk matched={bool(len(q) and q.risk_match.all())}")
    total=int(rep.transactions.sum())
    matched=int(rep.loc[rep.risk_match,"transactions"].sum())
    print(f"\nLocation values matched to country risk: {matched:,}/{total:,} ({matched/total:.2%})")
    miss=rep[(~rep.risk_match)&rep.country_iso2_key.notna()].sort_values("transactions",ascending=False)
    print("\nTop unmatched normalized countries:")
    print(miss.head(20).to_string(index=False) if len(miss) else "None")
    print(f"\nSaved audit: {a.out}")
    print("Raw inputs were not modified. HOLDOUT was not accessed.")

if __name__=="__main__":main()
