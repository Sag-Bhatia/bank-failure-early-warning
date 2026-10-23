"""Build preregistered features, public dates, labels and split (see FEATURES.md) -> data/features.parquet."""
import numpy as np
import pandas as pd

from fetch import DATA

TRAIN_END = pd.Timestamp("2018-03-31")
TEST_START, TEST_END = pd.Timestamp("2019-03-31"), pd.Timestamp("2024-12-31")
LAG = pd.Timedelta(days=45)
HORIZON = pd.Timedelta(days=365)

MODEL_A = ["eq_assets", "leverage", "npl_oreo", "texas", "roa", "cre_cap", "constr_cap",
           "loan_growth", "brokered", "nco", "log_assets"]
MODEL_B = MODEL_A + ["unreal_loss_t1", "uninsured", "dep_growth", "sec_assets", "cash_assets"]


def failures():
    f = pd.read_parquet(DATA / "failures.parquet")
    f = f[f.RESTYPE == "FAILURE"].assign(faildate=lambda d: pd.to_datetime(d.FAILDATE, format="%m/%d/%Y"))
    return f.sort_values("faildate").drop_duplicates("CERT")[["CERT", "NAME", "faildate"]]


def per_capital(num, cap):
    # capital at or below zero would otherwise become NaN and be imputed as "median"; floor at $1k so it scores as extreme
    return num / cap.clip(lower=1)


def per(num, den):
    return num / den.where(den > 0)


def build(fin, fails):
    d = fin.copy()
    d["repdte"] = pd.to_datetime(d.REPDTE, format="%Y%m%d")
    d["public_date"] = d.repdte + LAG
    num = d.NCLNLS + d.ORE
    prior = d[["CERT", "repdte", "LNLSGR", "DEP"]].assign(repdte=lambda x: x.repdte + pd.DateOffset(years=1))
    d = d.merge(prior, on=["CERT", "repdte"], how="left", suffixes=("", "_lag4"))
    quarter = d.repdte.dt.month // 3
    d = d.assign(
        eq_assets=per(d.EQ, d.ASSET),
        leverage=d.RBC1AAJ,
        npl_oreo=per(num, d.LNLSGR),
        texas=per_capital(num, d.EQ - d.INTAN + d.LNATRES),
        roa=d.ROA,
        cre_cap=per_capital(d.LNRENRES + d.LNREMULT + d.LNRECONS, d.RBCT1J),
        constr_cap=per_capital(d.LNRECONS, d.RBCT1J),
        loan_growth=per(d.LNLSGR, d.LNLSGR_lag4) - 1,
        brokered=per(d.BRO, d.DEP),
        nco=per(d.NTLNLS * 4 / quarter, d.LNLSGR),  # NTLNLS is year-to-date
        log_assets=np.log(d.ASSET.where(d.ASSET > 0)),
        unreal_loss_t1=per_capital((d.SCHA - d.SCHF) + (d.SCAA - d.SCAF), d.RBCT1J),
        uninsured=per(d.DEPUNINS, d.DEP),
        dep_growth=per(d.DEP, d.DEP_lag4) - 1,
        sec_assets=per(d.SC, d.ASSET),
        cash_assets=per(d.CHBAL, d.ASSET),
    )
    d = d.merge(fails[["CERT", "faildate"]], on="CERT", how="left")
    d = d[~(d.public_date >= d.faildate)]  # report not public before the bank failed
    d["y"] = ((d.faildate > d.public_date) & (d.faildate <= d.public_date + HORIZON)).astype(int)
    d["split"] = np.select([d.repdte <= TRAIN_END, d.repdte.between(TEST_START, TEST_END)], ["train", "test"], None)
    return d[["CERT", "NAME", "CITY", "STALP", "repdte", "public_date", "faildate", "y", "split", *MODEL_B]]


if __name__ == "__main__":
    out = build(pd.read_parquet(DATA / "financials.parquet"), failures())
    out.to_parquet(DATA / "features.parquet")
    print(out.groupby("split").agg(rows=("y", "size"), pos=("y", "sum"), banks_failed=("faildate", lambda s: s[out.loc[s.index, "y"] == 1].nunique())))
