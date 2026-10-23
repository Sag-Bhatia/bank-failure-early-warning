"""v3 data: BankFind 1985Q1-2024Q4 (one parquet per quarter, resumable), FRED yields, Fed BTFP/discount-window files."""
import time

import pandas as pd
import requests

from fetch import DATA, get, quarters

V3 = DATA / "v3"
FIELDS = ",".join("""CERT REPDTE RSSDID NAME CITY STALP ASSET DEP DEPUNINS EQ INTAN RBCT1J RBC1AAJ IDT1RWAJR
NCLNLS ORE LNLSGR P3ASSET NTLNLS LNATRES ROA NIMY EEFF CHBAL FREPO SC SCPLEDGE SCHA SCHF SCAA SCAF
OTHBOR FREPP BRO DEPLGAMT DEPLGB TRNNIA LNRENRES LNREMULT LNRECONS RTNVS
LNRS3LES LNRS3T12 LNRS1T3 LNRS3T5 LNRS5T15 LNRSOV15 LNOT3LES LNOT3T12 LNOT1T3 LNOT3T5 LNOT5T15 LNOTOV15
SCNM5T15 SCNMOV15 SCPT5T15 SCPTOV15""".split())
FRED = ["DGS3MO", "DGS1", "DGS3", "DGS5", "DGS10"]
FED = {
    "btfp.xlsx": "https://www.federalreserve.gov/publications/files/section-11s-btfp-transaction-specific-disclosures-03-12-25.xlsx",
    **{f"dw_{y}_q{q}.xlsx": f"https://www.federalreserve.gov/regreform/files/dw_data_{y}_q{q}.xlsx" for y in (2019, 2023) for q in (1, 2)},
}


def financials(first="1985-03-31", last="2024-12-31"):
    out = V3 / "q"
    out.mkdir(parents=True, exist_ok=True)
    for q in quarters(first, last):
        f = out / f"{q}.parquet"
        if f.exists():
            continue
        for attempt in range(4):
            try:
                df = get("financials", filters=f"REPDTE:{q}", fields=FIELDS)
                break
            except requests.RequestException as e:
                print(q, "retry", attempt, e, flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"failed {q}")
        num = df.columns.difference(["REPDTE", "NAME", "CITY", "STALP", "ID"])
        df[num] = df[num].apply(pd.to_numeric, errors="coerce").astype("float64")
        df.drop(columns="ID", errors="ignore").to_parquet(f)
        print(q, len(df), flush=True)


def yields():
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": ",".join(FRED)}, timeout=120)
    r.raise_for_status()
    from io import StringIO
    y = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"], na_values=".")
    y = y.set_index("observation_date").resample("QE").mean()
    y.to_parquet(V3 / "yields.parquet")
    print("yields", y.index.min().date(), y.index.max().date())


def fed_files():
    d = V3 / "fed"
    d.mkdir(parents=True, exist_ok=True)
    for name, url in FED.items():
        if not (d / name).exists():
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            (d / name).write_bytes(r.content)
        print(name, (d / name).stat().st_size)


if __name__ == "__main__":
    import sys
    V3.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 3:  # extra worker over a quarter range
        financials(*sys.argv[1:])
    else:
        yields()
        fed_files()
        financials()
