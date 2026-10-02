"""Canonical country normalization for SAML-D / country-risk enrichment."""
import re

ALIASES = {
    "US":"USA","USA":"USA","UNITED STATES":"USA","UNITED STATES OF AMERICA":"USA",
    "U S A":"USA","U.S.":"USA","U.S.A.":"USA",
    "UK":"GBR","GB":"GBR","GBR":"GBR","UNITED KINGDOM":"GBR",
    "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND":"GBR","GREAT BRITAIN":"GBR",
    "UAE":"ARE","AE":"ARE","ARE":"ARE","UNITED ARAB EMIRATES":"ARE",
}
def country_key(value):
    if value is None:
        return None
    s=str(value).strip().upper()
    if not s or s in {"NAN","NONE","NULL"}:
        return None
    cleaned=re.sub(r"[^A-Z0-9]+"," ",s).strip()
    return ALIASES.get(s,ALIASES.get(cleaned,cleaned))
