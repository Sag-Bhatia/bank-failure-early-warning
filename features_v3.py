"""v3 features, dimension scores and labels, exactly as PREREG_v3.md -> data/v3/features.parquet."""
import numpy as np
import pandas as pd

from features import HORIZON, LAG, failures, per, per_capital
from fetch_v3 import V3

# +1 higher = riskier, -1 higher = safer, 0 unconstrained (PREREG_v3.md)
SIGNS = {
    "leverage": -1, "tier1_rbc": -1, "eq_assets": -1, "mtm_cap": -1,
    "npl_oreo": 1, "pastdue": 1, "nco": 1, "reserve_cover": -1, "texas_reg": 1,
    "roa": -1, "nim": -1, "efficiency": 1,
    "liquid": -1, "run_gap": 1, "wholesale": 1, "brokered": 1, "loans_deposits": 1,
    "unreal_loss_t1": 1, "loan_mtm_t1": 1, "long_assets_t1": 1,
    "uninsured": 1, "large_acct_avg": 1, "big_operating": 1, "dep_growth": 0,
    "loan_growth": 1, "cre_cap": 1, "constr_cap": 1, "log_assets": 0, "swaps_assets": 0,
}
FEATURES = list(SIGNS)
DIMENSIONS = {
    "Capital": ["leverage", "tier1_rbc", "eq_assets", "mtm_cap"],
    "Asset quality": ["npl_oreo", "pastdue", "nco", "reserve_cover", "texas_reg"],
    "Earnings": ["roa", "nim", "efficiency"],
    "Liquidity": ["liquid", "run_gap", "wholesale", "brokered", "loans_deposits"],
    "Rate sensitivity": ["unreal_loss_t1", "loan_mtm_t1", "long_assets_t1"],
    "Depositors": ["uninsured", "large_acct_avg", "big_operating"],
    "Growth & concentration": ["loan_growth", "cre_cap", "constr_cap"],
}
# (bucket suffix, effective duration, FRED series) for the loan MTM estimate
BUCKETS = [("3LES", 0.1, "DGS3MO"), ("3T12", 0.5, "DGS1"), ("1T3", 1.8, "DGS3"),
           ("3T5", 3.5, "DGS5"), ("5T15", 6.0, "DGS10"), ("OV15", 7.0, "DGS10")]
COVERAGE_CHANGE = ["2008-12-31", "2009-03-31", "2010-09-30", "2010-12-31", "2011-03-31", "2013-03-31"]


def load_raw(fixed=False):
    parts = [pd.read_parquet(f) for f in sorted((V3 / "q").glob("*.parquet"))]
    raw = pd.concat(parts, ignore_index=True)
    if fixed:
        ieff = pd.concat([pd.read_parquet(f) for f in sorted((V3 / "ieff").glob("*.parquet"))], ignore_index=True)
        raw = raw.merge(ieff, on=["CERT", "REPDTE"], how="left")
    raw["repdte"] = pd.to_datetime(raw.REPDTE, format="%Y%m%d")
    return raw


def rate_rise(yields):
    """Rise in each Treasury yield over the past 8 quarters, by quarter end (percentage points)."""
    return (yields - yields.shift(8)).rename(columns=lambda c: f"dy_{c}")


def build(raw, yields, fails, fixed=False):
    """fixed=True is v3.1: the two unit bugs found after the v3 run (see PREREG_v3.md, v3.1 addendum)."""
    d = raw.copy()
    d["public_date"] = d.repdte + LAG
    dy = rate_rise(yields)
    dy.index = dy.index.normalize()
    d = d.merge(dy, left_on="repdte", right_index=True, how="left")
    lag = d[["CERT", "repdte", "LNLSGR", "DEP"]].assign(repdte=lambda x: x.repdte + pd.DateOffset(years=1))
    d = d.merge(lag, on=["CERT", "repdte"], how="left", suffixes=("", "_lag4"))

    loss = pd.Series(0.0, index=d.index)
    have = pd.Series(False, index=d.index)
    for suffix, dur, series in BUCKETS:
        amt = d[[f"LNRS{suffix}", f"LNOT{suffix}"]].sum(axis=1, min_count=1)
        have |= amt.notna()
        loss += amt.fillna(0) * dur * d[f"dy_{series}"].clip(lower=0) / 100
    loan_loss = loss.where(have)
    htm_loss = d.SCHA - d.SCHF
    afs_loss = d.SCAA - d.SCAF
    long_assets = d[["LNRS5T15", "LNRSOV15", "LNOT5T15", "LNOTOV15", "SCNM5T15", "SCNMOV15", "SCPT5T15", "SCPTOV15"]].sum(axis=1, min_count=1)
    liquid = d.CHBAL + d.FREPO.fillna(0) + (d.SC - d.SCPLEDGE.fillna(0)).clip(lower=0)
    quarter = d.repdte.dt.month // 3
    num = d.NCLNLS + d.ORE

    f = pd.DataFrame({
        "leverage": d.RBC1AAJ,
        # v3.1: 0 means "not reported" (pre-1990, and community-bank-leverage-ratio electors from 2020)
        "tier1_rbc": d.IDT1RWAJR.where(d.IDT1RWAJR > 0) if fixed else d.IDT1RWAJR,
        "eq_assets": per(d.EQ, d.ASSET),
        "mtm_cap": per(d.EQ - htm_loss - loan_loss, d.ASSET),
        "npl_oreo": per(num, d.LNLSGR),
        "pastdue": per(d.P3ASSET, d.ASSET),
        "nco": per(d.NTLNLS * 4 / quarter, d.LNLSGR),
        "reserve_cover": per(d.LNATRES, d.NCLNLS),
        "texas_reg": per_capital(num, d.RBCT1J + d.LNATRES),
        "roa": d.ROA,
        "nim": d.NIMY,
        "efficiency": per(d.EEFF, d.IEFF) if fixed else d.EEFF,  # v3 used the dollar numerator by mistake
        "liquid": per(liquid, d.ASSET),
        "run_gap": d.DEPUNINS / liquid.clip(lower=1),
        "wholesale": per(d.OTHBOR.fillna(0) + d.FREPP.fillna(0), d.ASSET),
        "brokered": per(d.BRO, d.DEP),
        "loans_deposits": per(d.LNLSGR, d.DEP),
        "unreal_loss_t1": per_capital(htm_loss.fillna(0) + afs_loss.fillna(0), d.RBCT1J).where(htm_loss.notna() | afs_loss.notna()),
        "loan_mtm_t1": per_capital(loan_loss, d.RBCT1J),
        "long_assets_t1": per_capital(long_assets, d.RBCT1J),
        "uninsured": per(d.DEPUNINS, d.DEP),
        "large_acct_avg": per(d.DEPLGAMT, d.DEPLGB) / 1000,  # $ thousands per account -> $ millions
        "big_operating": per(d.TRNNIA, d.DEP),
        "dep_growth": per(d.DEP, d.DEP_lag4) - 1,
        "loan_growth": per(d.LNLSGR, d.LNLSGR_lag4) - 1,
        "cre_cap": per_capital(d.LNRENRES + d.LNREMULT + d.LNRECONS, d.RBCT1J),
        "constr_cap": per_capital(d.LNRECONS, d.RBCT1J),
        "log_assets": np.log(d.ASSET.where(d.ASSET > 0)),
        "swaps_assets": per(d.RTNVS, d.ASSET),
    }, index=d.index).replace([np.inf, -np.inf], np.nan)
    out = pd.concat([d[["CERT", "RSSDID", "NAME", "CITY", "STALP", "ASSET", "DEPUNINS", "DEP", "repdte", "public_date"]], f], axis=1)

    fd = fails.set_index("CERT").faildate
    out["faildate"] = out.CERT.map(fd)
    out = out[~(out.public_date >= out.faildate)].copy()
    out["fail_365"] = ((out.faildate > out.public_date) & (out.faildate <= out.public_date + HORIZON)).astype(int)

    # run label: next quarter's uninsured deposits fell >= 20%, or the bank failed before filing it
    nxt = out[["CERT", "repdte", "DEPUNINS"]].assign(repdte=lambda x: x.repdte - pd.offsets.QuarterEnd(1))
    out = out.merge(nxt, on=["CERT", "repdte"], how="left", suffixes=("", "_next"))
    out["outcome_q"] = out.repdte + pd.offsets.QuarterEnd(1)
    failed_first = out.DEPUNINS_next.isna() & (out.faildate <= out.outcome_q + LAG)
    out["unins_chg_next"] = out.DEPUNINS_next / out.DEPUNINS - 1
    eligible = (out.uninsured >= 0.10) & (out.DEPUNINS_next.notna() | failed_first)
    out["run_event"] = np.where(eligible, ((out.unins_chg_next <= -0.20) | failed_first).astype(float), np.nan)
    return out


def dimension_scores(d):
    """Mean within-quarter percentile of each dimension's signed features; 1 = most fragile."""
    g = d.groupby("repdte")
    pct = {c: g[c].rank(pct=True, ascending=SIGNS[c] > 0) for c in FEATURES if SIGNS[c] != 0}
    return pd.DataFrame({dim: pd.concat([pct[c] for c in cols], axis=1).mean(axis=1) for dim, cols in DIMENSIONS.items()})


if __name__ == "__main__":
    import sys
    fixed = sys.argv[1:] == ["v31"]
    raw = load_raw(fixed)
    out = build(raw, pd.read_parquet(V3 / "yields.parquet"), failures(), fixed)
    out = pd.concat([out, dimension_scores(out)], axis=1)
    out.to_parquet(V3 / ("features_v31.parquet" if fixed else "features.parquet"))
    print(len(out), "rows;", out.repdte.min().date(), "to", out.repdte.max().date())
    print("coverage by decade:\n", out.groupby(out.repdte.dt.year // 10 * 10)[FEATURES].apply(lambda x: x.notna().mean()).T.round(2))
