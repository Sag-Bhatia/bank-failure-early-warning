"""v3.1 bug fix: the efficiency ratio needs IEFF (its income denominator); fetch it for every v3 quarter."""
import sys

import pandas as pd

from fetch import get, quarters
from fetch_v3 import V3

out = V3 / "ieff"
out.mkdir(parents=True, exist_ok=True)
for q in quarters(*sys.argv[1:3]):
    f = out / f"{q}.parquet"
    if not f.exists():
        df = get("financials", filters=f"REPDTE:{q}", fields="CERT,REPDTE,IEFF")
        df[["CERT", "IEFF"]] = df[["CERT", "IEFF"]].apply(pd.to_numeric, errors="coerce").astype("float64")
        df.drop(columns="ID", errors="ignore").to_parquet(f)
        print(q, len(df), flush=True)
