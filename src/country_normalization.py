"""Canonical ISO-2 country normalization for SAML-D / country-risk enrichment.

Uses pycountry for the full ISO 3166 country universe, with explicit aliases
for common banking/geographic labels (e.g. UK, USA, UAE).
"""
import re
import pycountry

ALIASES = {
    "UK":"GB","UAE":"AE",
    "USA":"US","U S A":"US","U.S.":"US","U.S.A.":"US",
    "UNITED STATES OF AMERICA":"US",
    "GREAT BRITAIN":"GB",
    "TURKEY":"TR",
}

def _clean(value):
    return re.sub(r"[^A-Z0-9]+"," ",str(value).strip().upper()).strip()

# Build lookup for ALL ISO countries: alpha-2, alpha-3, official/common names.
_LOOKUP={}
for c in pycountry.countries:
    _LOOKUP[c.alpha_2.upper()]=c.alpha_2.upper()
    _LOOKUP[c.alpha_3.upper()]=c.alpha_2.upper()
    _LOOKUP[_clean(c.name)]=c.alpha_2.upper()
    if hasattr(c,"official_name"):
        _LOOKUP[_clean(c.official_name)]=c.alpha_2.upper()
    if hasattr(c,"common_name"):
        _LOOKUP[_clean(c.common_name)]=c.alpha_2.upper()

for raw,iso2 in ALIASES.items():
    _LOOKUP[_clean(raw)]=iso2

def country_key(value):
    """Return canonical ISO-2 code, or None when the value cannot be resolved."""
    if value is None:
        return None
    s=str(value).strip()
    if not s or s.upper() in {"NAN","NONE","NULL"}:
        return None
    return _LOOKUP.get(_clean(s))
