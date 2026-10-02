"""Canonical ISO-2 country normalization for SAML-D / country-risk enrichment."""
import re

ALIASES = {
    "US":"US","USA":"US","UNITED STATES":"US","UNITED STATES OF AMERICA":"US",
    "U S A":"US","U.S.":"US","U.S.A.":"US",
    "UK":"GB","GB":"GB","GBR":"GB","UNITED KINGDOM":"GB",
    "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND":"GB","GREAT BRITAIN":"GB",
    "UAE":"AE","AE":"AE","ARE":"AE","UNITED ARAB EMIRATES":"AE",
}
def country_key(value):
    if value is None:
        return None
    s=str(value).strip().upper()
    if not s or s in {"NAN","NONE","NULL"}:
        return None
    cleaned=re.sub(r"[^A-Z0-9]+"," ",s).strip()
    return ALIASES.get(s,ALIASES.get(cleaned,cleaned))
