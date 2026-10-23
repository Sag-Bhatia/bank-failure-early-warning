"""Zero-parameter credit and run lenses (PREREG_v2.md). Percentile within quarter, 1 = most fragile."""
import pandas as pd

from features import per, per_capital


def lenses(fin):
    d = fin.copy()
    pledged = d.SCPLEDGE.fillna(0) if "SCPLEDGE" in d else 0
    d["texas"] = per_capital(d.NCLNLS + d.ORE, d.EQ - d.INTAN + d.LNATRES)
    d["mtm_cap"] = per(d.EQ - (d.SCHA - d.SCHF), d.ASSET)
    liquid = d.CHBAL + (d.SCAF + d.SCHF - pledged).clip(lower=0)
    d["run_gap"] = d.DEPUNINS / liquid.clip(lower=1)
    g = d.groupby("REPDTE")
    d["credit"] = g.texas.rank(pct=True)
    # mean of two percentiles; NaN if either component is missing
    d["run"] = (g.mtm_cap.rank(pct=True, ascending=False) + g.run_gap.rank(pct=True)) / 2
    return d
