"""v4 features (PREREG_v4.md): v3.1 base + trajectory block + regional block + size-relative run label -> data/v3/features_v4.parquet."""
import numpy as np
import pandas as pd

from features import LAG, failures
from features_v3 import SIGNS, build, dimension_scores, load_raw
from fetch_v3 import V3

TRAJ = {"d_leverage": -1, "d_npl_oreo": 1, "d_roa": -1, "d_uninsured": 1, "d_liquid": -1, "d_mtm_cap": -1}
REGION = {"state_unemp": 1, "state_unemp_chg": 1, "state_hpi_chg": -1}
BASE = [c for c in SIGNS]
SIGNS_V4 = {**SIGNS, **TRAJ, **REGION}
BLOCKS = {"base": BASE, "base+T": BASE + list(TRAJ), "base+G": BASE + list(REGION), "base+T+G": BASE + list(TRAJ) + list(REGION)}


def trajectory(d):
    src = {c[2:]: c for c in TRAJ}  # d_leverage <- leverage
    lag = d[["CERT", "repdte", *src]].assign(repdte=lambda x: x.repdte + pd.DateOffset(years=1))
    m = d[["CERT", "repdte"]].merge(lag, on=["CERT", "repdte"], how="left")
    return pd.DataFrame({new: d[old].to_numpy() - m[old].to_numpy() for old, new in src.items()}, index=d.index)


def regional(d, states):
    s = states.sort_values(["STALP", "repdte"]).copy()
    g = s.groupby("STALP")
    s["state_unemp"] = s.unemp
    s["state_unemp_chg"] = s.unemp - g.unemp.shift(4)
    s["state_hpi_chg"] = s.hpi / g.hpi.shift(4) - 1
    s["repdte"] = s.repdte.dt.normalize()
    m = d[["STALP", "repdte"]].merge(s[["STALP", "repdte", *REGION]], on=["STALP", "repdte"], how="left")
    return m[list(REGION)].set_index(d.index)


def run_label_v4(d):
    """Worst 5% of next-quarter uninsured change among same-size-quintile peers that quarter, or failed before filing."""
    failed_first = d.DEPUNINS_next.isna() & (d.faildate <= d.outcome_q + LAG)
    eligible = (d.uninsured >= 0.10) & (d.DEPUNINS_next.notna() | failed_first)
    e = d[eligible]
    quint = e.groupby("repdte").log_assets.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    cut = e.unins_chg_next.groupby([e.repdte, quint]).transform(lambda x: x.quantile(0.05))
    event = ((e.unins_chg_next <= cut) | failed_first[eligible]).astype(float)
    return event.reindex(d.index)  # NaN where not eligible


if __name__ == "__main__":
    raw = load_raw(fixed=True)
    d = build(raw, pd.read_parquet(V3 / "yields.parquet"), failures(), fixed=True)
    d = pd.concat([d, trajectory(d), regional(d, pd.read_parquet(V3 / "states.parquet"))], axis=1)
    d["run_event_v4"] = run_label_v4(d)
    d = pd.concat([d, dimension_scores(d)], axis=1).replace([np.inf, -np.inf], np.nan)
    d.to_parquet(V3 / "features_v4.parquet")
    print(len(d), "rows", d.repdte.min().date(), "-", d.repdte.max().date())
    print("new-feature coverage:", d[list(TRAJ) + list(REGION)].notna().mean().round(2).to_dict())
    r = d[d.run_event_v4.notna()]
    print("run label v4 rate by size quintile:", r.groupby(pd.qcut(r.log_assets, 5, labels=False)).run_event_v4.mean().round(3).tolist(),
          "| v3 label:", r.groupby(pd.qcut(r.log_assets, 5, labels=False)).run_event.mean().round(3).tolist())
