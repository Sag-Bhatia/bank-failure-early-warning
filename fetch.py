"""Download quarterly financials and the failure list from the FDIC BankFind API into data/*.parquet."""
from pathlib import Path

import pandas as pd
import requests

API = "https://api.fdic.gov/banks"
DATA = Path(__file__).parent / "data"
FIELDS = ("CERT,REPDTE,NAME,CITY,STALP,ASSET,DEP,DEPUNINS,SC,SCHA,SCHF,SCAA,SCAF,CHBAL,EQ,INTAN,LNATRES,"
          "NCLNLS,ORE,LNLSGR,RBCT1J,RBC1AAJ,ROA,LNRECONS,LNRENRES,LNREMULT,BRO,NTLNLS")


def get(endpoint, **params):
    rows, offset = [], 0
    while True:
        r = requests.get(f"{API}/{endpoint}", params={**params, "limit": 10000, "offset": offset, "format": "json"}, timeout=120)
        r.raise_for_status()
        page = r.json()
        rows += [d["data"] for d in page["data"]]
        offset += 10000
        if offset >= page["meta"]["total"]:
            return pd.DataFrame(rows)


def quarters(first="2001-03-31", last="2024-12-31"):
    return [d.strftime("%Y%m%d") for d in pd.date_range(first, last, freq="QE")]


def main():
    DATA.mkdir(exist_ok=True)
    out = DATA / "financials.parquet"
    if not out.exists():
        parts = []
        for q in quarters():
            df = get("financials", filters=f"REPDTE:{q}", fields=FIELDS)
            print(q, len(df), flush=True)
            parts.append(df)
        pd.concat(parts, ignore_index=True).to_parquet(out)
    fails = get("failures", fields="CERT,NAME,FAILDATE,RESTYPE,RESTYPE1,COST,QBFASSET")
    fails.to_parquet(DATA / "failures.parquet")
    print("failures", len(fails))


if __name__ == "__main__":
    main()
