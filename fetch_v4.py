"""v4 data: BankFind 2025Q1-2026Q2 (same fields as v3, plus IEFF) and FRED state unemployment and house prices."""
import time
from io import StringIO

import pandas as pd
import requests

from fetch import get, quarters
from fetch_v3 import V3, financials

STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND "
          "OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()


def state_series():
    ids = [f"{s}UR" for s in STATES] + [f"{s}STHPI" for s in STATES]
    frames = []

    def fred(batch):
        for attempt in range(3):
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": ",".join(batch)}, timeout=180)
            if r.ok and r.text.startswith("observation_date"):
                return pd.read_csv(StringIO(r.text), parse_dates=["observation_date"], na_values=".").set_index("observation_date")
            time.sleep(5 * (attempt + 1))
        return None

    for i in range(0, len(ids), 10):
        batch = ids[i:i + 10]
        got = fred(batch)
        if got is None:  # fall back to one series at a time so a single bad ID can't sink the batch
            got = pd.concat([x for x in (fred([b]) for b in batch) if x is not None], axis=1)
        frames.append(got)
    wide = pd.concat(frames, axis=1).resample("QE").mean()
    print("missing series:", [c for c in ids if c not in wide])
    wide = wide.reindex(columns=ids)
    long = []
    for s in STATES:
        long.append(pd.DataFrame({"STALP": s, "repdte": wide.index, "unemp": wide[f"{s}UR"].values, "hpi": wide[f"{s}STHPI"].values}))
    out = pd.concat(long, ignore_index=True)
    out.to_parquet(V3 / "states.parquet")
    print("states", out.repdte.min().date(), out.repdte.max().date(), out.unemp.notna().mean().round(3), out.hpi.notna().mean().round(3))


def ieff(first, last):
    out = V3 / "ieff"
    for q in quarters(first, last):
        f = out / f"{q}.parquet"
        if not f.exists():
            df = get("financials", filters=f"REPDTE:{q}", fields="CERT,REPDTE,IEFF")
            df[["CERT", "IEFF"]] = df[["CERT", "IEFF"]].apply(pd.to_numeric, errors="coerce").astype("float64")
            df.drop(columns="ID", errors="ignore").to_parquet(f)


if __name__ == "__main__":
    state_series()
    financials("2025-03-31", "2026-06-30")
    ieff("2025-03-31", "2026-06-30")
